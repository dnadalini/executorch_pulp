/* Copyright (c) 2026 KU Leuven. */
#include "pulp_runtime_context.h"
#include "pmsis.h"

#include <stddef.h>
#include <stdint.h>

#ifndef NUM_CORES
#define NUM_CORES 8
#endif
#ifndef PULP_QUANTIZATION_L1_BYTES
#define PULP_QUANTIZATION_L1_BYTES 32768
#endif
#if PULP_QUANTIZATION_L1_BYTES < 256
#error "PULP_QUANTIZATION_L1_BYTES must be at least 256"
#endif

// This file is compiled for the cluster's FP32 hardware. FC wrappers and
// dispatch only pass the FP32 parameters; they never call these workers on FC.
typedef struct {
  float* floats;
  uint8_t* bytes;
  size_t count;
  float multiplier;
  int32_t zero_point;
  int32_t quant_min;
  int32_t quant_max;
  int quantize;
  int errors[NUM_CORES];
} QuantizationTile;

static void quantization_worker(void* context) {
  QuantizationTile* tile = (QuantizationTile*)context;
  const unsigned core = pi_core_id();
  // Whole four-byte output groups avoid different cores updating one byte word.
  const size_t chunk = ((tile->count + NUM_CORES - 1) / NUM_CORES + 3) & ~(size_t)3;
  const size_t begin = chunk * core;
  size_t end = begin + chunk;
  if (end > tile->count) end = tile->count;
  tile->errors[core] = 0;
  if (tile->quantize) {
    const float lower = (float)(tile->quant_min - tile->zero_point);
    const float upper = (float)(tile->quant_max - tile->zero_point);
    for (size_t i = begin; i < end; ++i) {
      const float value = tile->floats[i] * tile->multiplier;
      if (value != value) {
        tile->errors[core] = 1;
        continue;
      }
      if (value <= lower) {
        tile->bytes[i] = (uint8_t)tile->quant_min;
      } else if (value >= upper) {
        tile->bytes[i] = (uint8_t)tile->quant_max;
      } else {
        // Truncation plus explicit ties-to-even matches the existing FC kernel,
        // including negative halfway values with a nonzero zero point.
        int32_t rounded = (int32_t)value;
        if ((float)rounded > value) --rounded;
        const float fraction = value - (float)rounded;
        if (fraction > 0.5f || (fraction == 0.5f && (rounded & 1))) ++rounded;
        tile->bytes[i] = (uint8_t)(rounded + tile->zero_point);
      }
    }
  } else {
    for (size_t i = begin; i < end; ++i) {
      tile->floats[i] = ((int32_t)tile->bytes[i] - tile->zero_point) * tile->multiplier;
    }
  }
}

void pulp_cluster_quantization_entry(void* context) {
  PulpQuantizationArgs* args = (PulpQuantizationArgs*)context;
  if (args->count == 0) { args->status = 0; return; }
  const size_t overhead = (sizeof(QuantizationTile) + 7) & ~(size_t)7;
  if (PULP_QUANTIZATION_L1_BYTES < overhead + 20) {
    args->status = PULP_QUANTIZATION_RUNTIME_ERROR;
    return;
  }
  // One FP32 and one UINT8 buffer, rounded down to keep tile offsets aligned.
  size_t capacity = ((PULP_QUANTIZATION_L1_BYTES - overhead) / 5) & ~(size_t)3;
  // The DMA command size is bounded, even if the configured L1 budget is large.
  if (capacity > 16380) capacity = 16380;
  if (args->count < capacity) capacity = (args->count + 3) & ~(size_t)3;
  uint8_t* arena = NULL;
  size_t arena_bytes = 0;
  // Other cluster users/stacks can consume L1. Retry with smaller tiles instead
  // of assuming all nominal L1 memory is available or falling back to the FC.
  while (capacity >= 4) {
    arena_bytes = overhead + capacity * 5;
    arena = (uint8_t*)pi_cl_l1_malloc(0, arena_bytes);
    if (arena != NULL) break;
    capacity = (capacity / 2) & ~(size_t)3;
  }
  if (arena == NULL) {
    args->status = PULP_QUANTIZATION_RUNTIME_ERROR;
    return;
  }
  QuantizationTile* tile = (QuantizationTile*)arena;
  tile->floats = (float*)(arena + overhead);
  tile->bytes = arena + overhead + capacity * sizeof(float);
  tile->multiplier = args->quantize ? args->inverse_scale : args->scale;
  tile->zero_point = args->zero_point;
  tile->quant_min = args->quant_min;
  tile->quant_max = args->quant_max;
  tile->quantize = args->quantize;
  args->status = 0;
  for (size_t offset = 0; offset < args->count;) {
    size_t count = args->count - offset;
    if (count > capacity) count = capacity;
    tile->count = count;
    pi_cl_dma_cmd_t transfer;
    const size_t input_item_bytes = args->quantize ? sizeof(float) : 1;
    const size_t output_item_bytes = args->quantize ? 1 : sizeof(float);
    void* tile_input = args->quantize ? (void*)tile->floats : (void*)tile->bytes;
    void* tile_output = args->quantize ? (void*)tile->bytes : (void*)tile->floats;
    pi_cl_dma_cmd(
        (uintptr_t)((const uint8_t*)args->input + offset * input_item_bytes),
        (uintptr_t)tile_input, count * input_item_bytes,
        PI_CL_DMA_DIR_EXT2LOC, &transfer);
    pi_cl_dma_cmd_wait(&transfer);
    pi_cl_team_fork(NUM_CORES, quantization_worker, tile);
    for (unsigned core = 0; core < NUM_CORES; ++core) {
      if (tile->errors[core]) args->status = PULP_QUANTIZATION_INVALID_INPUT;
    }
    if (args->status != 0) break;
    pi_cl_dma_cmd(
        (uintptr_t)((uint8_t*)args->output + offset * output_item_bytes),
        (uintptr_t)tile_output, count * output_item_bytes,
        PI_CL_DMA_DIR_LOC2EXT, &transfer);
    pi_cl_dma_cmd_wait(&transfer);
    offset += count;
  }
  pi_cl_l1_free(0, arena, arena_bytes);
}
