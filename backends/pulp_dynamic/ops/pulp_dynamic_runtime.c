/* Copyright (c) 2026 KU Leuven. */
#include "pulp_dynamic_runtime.h"

#include <limits.h>
#include <stdarg.h>
#include <stdio.h>

#include "pmsis.h"
#include "pulp_runtime_context.h"
#include "solver.h"
#ifndef USE_DOUBLE_BUFFERING
#define USE_DOUBLE_BUFFERING 1
#endif
#ifndef PULP_DYNAMIC_VERBOSE
#define PULP_DYNAMIC_VERBOSE 0
#endif

#define LAYER_CONV2D 0
#define LAYER_POINTWISE 1
#define LAYER_DEPTHWISE 2
#define LAYER_KIND_MASK 3
#define LAYER_FUSED_RELU 4

volatile int pulp_dynamic_active_layer = -1;
static const char* active_phase = "idle";

typedef struct {
  uint32_t tiles, metadata, solver, execution, rows;
  int valid;
} LayerStats;

static LayerStats layer_stats[256];

void pulp_dynamic_report_stats(unsigned case_id, const int* instructions,
                               size_t count) {
  for (unsigned i = 0; i < 256; ++i)
    if (layer_stats[i].valid) {
      LayerStats* s = &layer_stats[i];
      printf(
          "DYNAMIC_LAYER case=%u instruction=%d layer_id=%u tiles=%u rows=%u "
          "metadata_bytes=%u solver_cluster_cycles=%u "
          "template_cluster_cycles=%u\n",
          case_id, i < count ? instructions[i] : -1, i, s->tiles, s->rows,
          s->metadata, s->solver, s->execution);
    }
}

int pulp_dynamic_double_buffering(void) { return USE_DOUBLE_BUFFERING; }

int pulp_dynamic_solver_printf(const char* format, ...) {
#if PULP_DYNAMIC_VERBOSE
  va_list args;
  va_start(args, format);
  int result = vprintf(format, args);
  va_end(args);
  return result;
#else
  (void)format;
  return 0;
#endif
}

void pulp_dynamic_solver_exit(int code) {
  printf("PULP_DYNAMIC_FAIL layer=%d phase=%s status=%d\n",
         pulp_dynamic_active_layer, active_phase, code);
  pmsis_exit(code ? code : -1);
  __builtin_unreachable();
}

static void cluster_entry(void* context) {
  PulpDynamicArgs* a = (PulpDynamicArgs*)context;
  const int kind = a->kind & LAYER_KIND_MASK;
  a->status = -1;
  // Both quantization arrays are transferred by the template even
  // when native batchnorm is disabled. They must be real L2 allocations.
  pi_cl_alloc_req_t allocation;
  const uint32_t qbytes = 2u * a->co * sizeof(int32_t);
  pi_cl_l2_malloc(qbytes, &allocation);
  int32_t* quant = (int32_t*)pi_cl_l2_malloc_wait(&allocation);
  if (!quant) return;
  for (uint32_t c = 0; c < a->co; ++c) {
    quant[c] = 1;
    quant[a->co + c] = 0;
  }
  profiler_heuristics_t heuristics = {0};
  heuristics.Ci_priority = 0;
  heuristics.Co_priority = 1;
  heuristics.Wo_priority = 2;
  heuristics.Ho_priority = 3;
  heuristics.Co_spatial_mapping = kind == LAYER_DEPTHWISE ? 1 : 4;
  heuristics.Ho_spatial_mapping = NUM_CORES;
  heuristics.Wo_spatial_mapping = kind == LAYER_DEPTHWISE ? 1 : 2;
  PULP_Conv2d_tiling_args_INT8 config = {0};
  solver_output_conv2d_int8_t output = {0};
  uint32_t count = 0;
  output.tiling_config_struct = &config;
  output.num_tiles = &count;
  solver_input_conv2d_int8_t input = {0};
  input.L2_input = (int8_t*)a->input;
  input.L2_weights = (int8_t*)a->weight;
  input.L2_bias = (int32_t*)a->bias;
  input.L2_output = a->output;
  input.Hi = a->hi;
  input.Wi = a->wi;
  input.Ci = a->ci;
  input.Hk = a->kh;
  input.Wk = a->kw;
  input.Ho = a->ho;
  input.Wo = a->wo;
  input.Co = a->co;
  input.padding_H = a->ph;
  input.padding_W = a->pw;
  input.stride_H = a->sh;
  input.stride_W = a->sw;
  input.has_bias = a->bias != NULL;
  const uint64_t scratch =
      kind == LAYER_POINTWISE
          ? 0
          : (kind == LAYER_DEPTHWISE
                 ? (uint64_t)NUM_CORES * a->kw * (a->hi + 2 * a->ph + 1)
                 : (uint64_t)2 * NUM_CORES * a->kh * a->kw * a->ci);
  const uint64_t bytes =
      (uint64_t)a->hi * a->wi * a->ci +
      (uint64_t)a->kh * a->kw * (kind == LAYER_DEPTHWISE ? 1 : a->ci) * a->co +
      (uint64_t)a->ho * a->wo * a->co + (a->bias ? 4u * a->co : 0) + qbytes +
      scratch + 32;
  if (bytes * (USE_DOUBLE_BUFFERING ? 2u : 1u) > UINT32_MAX) goto cleanup;
  // The solver compares this against the total arena before selecting its
  // per-buffer budget. Double templates reserve two operand slots even for
  // one tile, so use the footprint of both slots to avoid an invalid untiled
  // fast path when only a single copy of the full layer would fit.
  input.layer_memocc_B = (uint32_t)(bytes * (USE_DOUBLE_BUFFERING ? 2u : 1u));
  input.kappa = quant;
  input.lambda = quant + a->co;
  input.out_mult = 1;
  input.out_shift = a->shift;
  input.flag_relu = (a->kind & LAYER_FUSED_RELU) != 0;
  input.flag_batch_norm = false;
  input.conv2d_heuristics = &heuristics;
  input.available_memory_B = a->l1_bytes;
  input.use_im2col = kind != LAYER_POINTWISE;
  input.max_iterations = a->max_iterations;
  input.loop_impact_memory_limit_kb = 0;
  input.tiling_shape_strategy = a->spatial_mapping_multiple
                                    ? SPATIAL_MAPPING_MULTIPLE
                                    : MAX_L1_OCCUPATION;
  active_phase = "solver";
  pi_perf_conf(1 << PI_PERF_CYCLES);
  pi_perf_reset();
  pi_perf_start();
  on_device_solver(&input, &output, kind == LAYER_DEPTHWISE ? DW_CONV : CONV2D,
                   SATURATION, INT8);
  if (!config.numTiles || !config.numTiles[1]) goto cleanup;
  a->metadata_bytes = output.metadata_size_B;
  a->tiles = config.numTiles[1] - config.numTiles[0];
  a->configuration_cycles = pi_perf_read(PI_PERF_CYCLES);
  if ((uint64_t)a->metadata_bytes + qbytes > a->metadata_budget) {
    a->status = -3;
    goto cleanup;
  }
  config.activation_max_plus_one = a->activation_max_plus_one;
  active_phase = "template";
  // The same geometry is reused for linear rows within this invocation only.
  for (uint32_t row = 0; row < a->rows; ++row) {
    config.l2_input = (int8_t*)a->input + (size_t)row * a->hi * a->wi * a->ci;
    config.l2_output = a->output + (size_t)row * a->ho * a->wo * a->co;
#if USE_DOUBLE_BUFFERING
    if (kind == LAYER_DEPTHWISE)
      l2_double_buffering_template_depthwise_conv_int8(&config);
    else
      l2_double_buffering_template_conv2d_int8(&config);
#else
    if (kind == LAYER_DEPTHWISE)
      l2_single_buffering_template_depthwise_conv_int8(&config);
    else
      l2_single_buffering_template_conv2d_int8(&config);
#endif
  }
  a->execution_cycles = pi_perf_read(PI_PERF_CYCLES) - a->configuration_cycles;
  a->status = 0;
cleanup:
  active_phase = "cleanup";
  free_conv2d_solver_metadata_int8(&output);
  pi_cl_free_req_t release;
  pi_cl_l2_free(quant, qbytes, &release);
  pi_cl_l2_free_wait(&release);
  active_phase = "idle";
}

int pulp_dynamic_run(PulpDynamicArgs* args) {
  if (!args) return 0;
  args->status = -1;
  pulp_dynamic_active_layer = args->layer_index;
  if (!args->rows) {
    args->status = 0;
    return 1;
  }
  // Probe contiguous native-allocator capacity after ExecuTorch has allocated
  // its arenas. Release it immediately so the solver can use that headroom.
  void* headroom = pi_l2_malloc(args->metadata_budget);
  if (!headroom) {
    args->status = -3;
    printf("PULP_DYNAMIC_FAIL layer=%d phase=L2-headroom budget=%u\n",
           args->layer_index, args->metadata_budget);
    return 0;
  }
  pi_l2_free(headroom, args->metadata_budget);
  if (!pulp_runtime_run_task_with_stack(cluster_entry, args, 8192, 1024))
    return 0;
  if (args->status)
    printf("PULP_DYNAMIC_FAIL layer=%d status=%d metadata_bytes=%u\n",
           args->layer_index, args->status, args->metadata_bytes);
  if (args->status == 0 && args->layer_index >= 0 && args->layer_index < 256) {
    LayerStats s = {args->tiles,
                    args->metadata_bytes,
                    args->configuration_cycles,
                    args->execution_cycles,
                    args->rows,
                    1};
    layer_stats[args->layer_index] = s;
  }
  return args->status == 0;
}
