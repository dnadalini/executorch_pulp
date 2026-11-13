/* Copyright (c) 2026 KU Leuven. */
#include <math.h>
#include <stdint.h>

#include "pmsis.h"
#include "pulp_runtime_context.h"

#ifndef NUM_CORES
#define NUM_CORES 8
#endif
#ifndef PULP_ACTIVATION_L1_BYTES
#define PULP_ACTIVATION_L1_BYTES 32768
#endif

typedef struct {
  PulpActivationArgs job;
  void* input;
  void* output;
  size_t count;
  int errors[NUM_CORES];
} ActivationTile;

static void activation_worker(void* context) {
  ActivationTile* tile = (ActivationTile*)context;
  const PulpActivationArgs* job = &tile->job;
  const unsigned core = pi_core_id();
  const size_t chunk =
      ((tile->count + NUM_CORES - 1) / NUM_CORES + 3) & ~(size_t)3;
  const size_t begin = chunk * core;
  size_t end = begin + chunk;
  if (end > tile->count) end = tile->count;
  tile->errors[core] = 0;
  for (size_t i = begin; i < end; ++i) {
    float x = job->input_quantized
                  ? ((int32_t)((const uint8_t*)tile->input)[i] -
                     job->input_zero_point) *
                        job->input_scale
                  : ((const float*)tile->input)[i];
    if (job->kind <= 1) {
      // Comparisons preserve NaN, as ATen clamp/ReLU do.
      if (x < job->lower) x = job->lower;
      if (x > job->upper) x = job->upper;
    } else if (job->kind == 2) {
      x = (0.5f * x) * (1.0f + erff(x * 0.7071067811865475f));
    } else {
      const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
      x = (0.5f * x) * (1.0f + tanhf(inner));
    }
    if (!job->output_quantized) {
      ((float*)tile->output)[i] = x;
      continue;
    }
    x *= job->output_inverse_scale;
    if (x != x) {
      tile->errors[core] = 1;
      continue;
    }
    const float lower = (float)(job->quant_min - job->output_zero_point);
    const float upper = (float)(job->quant_max - job->output_zero_point);
    int32_t q;
    if (x <= lower)
      q = job->quant_min;
    else if (x >= upper)
      q = job->quant_max;
    else {
      q = (int32_t)x;
      if ((float)q > x) --q;
      const float fraction = x - (float)q;
      if (fraction > 0.5f || (fraction == 0.5f && (q & 1))) ++q;
      q += job->output_zero_point;
    }
    ((uint8_t*)tile->output)[i] = (uint8_t)q;
  }
}

void pulp_cluster_activation_entry(void* context) {
  PulpActivationArgs* args = (PulpActivationArgs*)context;
  args->status = PULP_QUANTIZATION_RUNTIME_ERROR;
  if (args->count == 0) {
    args->status = 0;
    return;
  }
  const size_t input_bytes = args->input_quantized ? 1 : sizeof(float);
  const size_t output_bytes = args->output_quantized ? 1 : sizeof(float);
  const size_t overhead = (sizeof(ActivationTile) + 7) & ~(size_t)7;
  if (PULP_ACTIVATION_L1_BYTES < overhead + 4 * (input_bytes + output_bytes))
    return;
  size_t capacity =
      ((PULP_ACTIVATION_L1_BYTES - overhead) / (input_bytes + output_bytes)) &
      ~(size_t)3;
  if (capacity > 16380) capacity = 16380;  // Every DMA command fits in 16 bits.
  if (args->count < capacity) capacity = (args->count + 3) & ~(size_t)3;
  uint8_t* arena = NULL;
  size_t arena_bytes = 0;
  while (capacity >= 4) {
    arena_bytes = overhead + capacity * (input_bytes + output_bytes);
    arena = (uint8_t*)pi_cl_l1_malloc(0, arena_bytes);
    if (arena != NULL) break;
    capacity = (capacity / 2) & ~(size_t)3;
  }
  if (arena == NULL) return;
  ActivationTile* tile = (ActivationTile*)arena;
  tile->job = *args;
  tile->input = arena + overhead;
  tile->output = arena + overhead + capacity * input_bytes;
  args->status = 0;
  for (size_t offset = 0; offset < args->count;) {
    size_t count = args->count - offset;
    if (count > capacity) count = capacity;
    tile->count = count;
    pi_cl_dma_cmd_t transfer;
    pi_cl_dma_cmd(
        (uintptr_t)((const uint8_t*)args->input + offset * input_bytes),
        (uintptr_t)tile->input, count * input_bytes, PI_CL_DMA_DIR_EXT2LOC,
        &transfer);
    pi_cl_dma_cmd_wait(&transfer);
    pi_cl_team_fork(NUM_CORES, activation_worker, tile);
    for (unsigned core = 0; core < NUM_CORES; ++core)
      if (tile->errors[core]) args->status = PULP_QUANTIZATION_INVALID_INPUT;
    if (args->status != 0) break;
    pi_cl_dma_cmd((uintptr_t)((uint8_t*)args->output + offset * output_bytes),
                  (uintptr_t)tile->output, count * output_bytes,
                  PI_CL_DMA_DIR_LOC2EXT, &transfer);
    pi_cl_dma_cmd_wait(&transfer);
    offset += count;
  }
  pi_cl_l1_free(0, arena, arena_bytes);
}
