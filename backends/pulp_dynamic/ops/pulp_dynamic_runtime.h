/* Copyright (c) 2026 KU Leuven. */
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const int8_t *input, *weight;
  const int32_t* bias;
  int8_t* output;
  uint32_t hi, wi, ci, ho, wo, co, kh, kw, sh, sw, ph, pw, rows;
  uint32_t l1_bytes, metadata_budget, max_iterations;
  int shift, kind, layer_index, spatial_mapping_multiple, status;
  uint32_t activation_max_plus_one;
  uint32_t metadata_bytes, tiles, configuration_cycles, execution_cycles;
} PulpDynamicArgs;
int pulp_dynamic_run(PulpDynamicArgs* args);
int pulp_dynamic_double_buffering(void);
// Runner calls this only after FC performance counters have stopped.
void pulp_dynamic_report_stats(
    unsigned case_id,
    const int* instructions,
    size_t count);
extern volatile int pulp_dynamic_active_layer;

#ifdef __cplusplus
}
#endif
