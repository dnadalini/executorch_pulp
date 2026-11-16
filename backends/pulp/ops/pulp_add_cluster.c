/* Copyright (c) 2026 KU Leuven. */
#include "pulp_dma.h"
#include "pulp_runtime_context.h"

#ifndef NUM_CORES
#define NUM_CORES 8
#endif
#ifndef PULP_ADD_L1_BYTES
#define PULP_ADD_L1_BYTES 32768
#endif

typedef struct {
  float* left;
  float* right;
  size_t count;
  float alpha;
} AddTile;

static void add_worker(void* context) {
  AddTile* tile = (AddTile*)context;
  // Stripe adjacent FP32 elements across cores, avoiding the bank conflicts
  // caused by cores starting at equal power-of-two offsets in L1.
  const size_t begin = pi_core_id();
  if (tile->alpha == 1.0f) {
    for (size_t i = begin; i < tile->count; i += NUM_CORES)
      tile->left[i] += tile->right[i];
  } else {
    for (size_t i = begin; i < tile->count; i += NUM_CORES) {
      const float scaled = tile->alpha * tile->right[i];
      tile->left[i] += scaled;
    }
  }
}

void pulp_cluster_add_entry(void* context) {
  PulpAddArgs* args = (PulpAddArgs*)context;
  args->status = -1;
  if (!args->count) {
    args->status = 0;
    return;
  }
  const size_t overhead = (sizeof(AddTile) + 7) & ~(size_t)7;
  if (PULP_ADD_L1_BYTES < overhead + 8 * sizeof(float))
    return;
  // Two FP32 input buffers; the left buffer is reused for the output.
  size_t capacity =
      ((PULP_ADD_L1_BYTES - overhead) / (2 * sizeof(float))) & ~(size_t)3;
  if (capacity > 16380)
    capacity = 16380;
  if (args->count < capacity)
    capacity = (args->count + 3) & ~(size_t)3;
  uint8_t* arena = NULL;
  size_t arena_bytes = 0;
  while (capacity >= 4) {
    arena_bytes = overhead + 2 * capacity * sizeof(float);
    arena = (uint8_t*)pi_cl_l1_malloc(0, arena_bytes);
    if (arena)
      break;
    capacity = (capacity / 2) & ~(size_t)3;
  }
  if (!arena)
    return;
  AddTile* tile = (AddTile*)arena;
  tile->left = (float*)(arena + overhead);
  tile->right = tile->left + capacity;
  tile->alpha = args->alpha;
  for (size_t offset = 0; offset < args->count;) {
    size_t count = args->count - offset;
    if (count > capacity)
      count = capacity;
    tile->count = count;
    pulp_dma_copy_in(tile->left, args->left + offset, count * sizeof(float));
    pulp_dma_copy_in(tile->right, args->right + offset, count * sizeof(float));
    pi_cl_team_fork(NUM_CORES, add_worker, tile);
    pulp_dma_copy_out(args->output + offset, tile->left, count * sizeof(float));
    offset += count;
  }
  pi_cl_l1_free(0, arena, arena_bytes);
  args->status = 0;
}
