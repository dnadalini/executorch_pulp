/* Copyright (c) 2026 KU Leuven. */
#include "pmsis.h"
#include "pulp_runtime_context.h"

#ifndef NUM_CORES
#define NUM_CORES 8
#endif
#ifndef PULP_GLOBAL_POOL_L1_BYTES
#define PULP_GLOBAL_POOL_L1_BYTES 32768
#endif

typedef struct {
  float* input;
  float* sums;
  size_t channels;
  size_t pixels;
  float inverse_spatial;
  int channels_last;
  int last;
} PoolTile;

static size_t smaller(size_t a, size_t b) { return a < b ? a : b; }

static void pooling_worker(void* context) {
  PoolTile* tile = (PoolTile*)context;
  // Each channel has exactly one owner, also across successive spatial chunks.
  for (size_t c = pi_core_id(); c < tile->channels; c += NUM_CORES) {
    float sum = tile->sums[c];
    if (tile->channels_last) {
      for (size_t p = 0; p < tile->pixels; ++p)
        sum += tile->input[p * tile->channels + c];
    } else {
      for (size_t p = 0; p < tile->pixels; ++p)
        sum += tile->input[c * tile->pixels + p];
    }
    tile->sums[c] = tile->last ? sum * tile->inverse_spatial : sum;
  }
}

static void load_tile(
    const PulpGlobalPoolArgs* args,
    PoolTile* tile,
    size_t batch,
    size_t channel,
    size_t pixel) {
  const size_t channels = tile->channels, pixels = tile->pixels;
  const float* input = args->input + batch * args->channels * args->spatial;
  pi_cl_dma_cmd_t transfer;
  if (!args->channels_last && pixels == args->spatial) {
    pi_cl_dma_cmd(
        (uintptr_t)(input + channel * args->spatial),
        (uintptr_t)tile->input,
        channels * pixels * sizeof(float),
        PI_CL_DMA_DIR_EXT2LOC,
        &transfer);
    pi_cl_dma_cmd_wait(&transfer);
  } else if (!args->channels_last) {
    for (size_t c = 0; c < channels; ++c) {
      pi_cl_dma_cmd(
          (uintptr_t)(input + (channel + c) * args->spatial + pixel),
          (uintptr_t)(tile->input + c * pixels),
          pixels * sizeof(float),
          PI_CL_DMA_DIR_EXT2LOC,
          &transfer);
      pi_cl_dma_cmd_wait(&transfer);
    }
  } else if (channels == args->channels) {
    pi_cl_dma_cmd(
        (uintptr_t)(input + pixel * args->channels),
        (uintptr_t)tile->input,
        channels * pixels * sizeof(float),
        PI_CL_DMA_DIR_EXT2LOC,
        &transfer);
    pi_cl_dma_cmd_wait(&transfer);
  } else if (args->channels <= 65535 / sizeof(float)) {
    // Gather a channel range from HWC without an L2 layout conversion.
    pi_cl_dma_cmd_2d(
        (uintptr_t)(input + pixel * args->channels + channel),
        (uintptr_t)tile->input,
        channels * pixels * sizeof(float),
        args->channels * sizeof(float),
        channels * sizeof(float),
        PI_CL_DMA_DIR_EXT2LOC,
        &transfer);
    pi_cl_dma_cmd_wait(&transfer);
  } else {
    // DMA strides, as well as lengths, have a 16-bit hardware limit.
    for (size_t p = 0; p < pixels; ++p) {
      pi_cl_dma_cmd(
          (uintptr_t)(input + (pixel + p) * args->channels + channel),
          (uintptr_t)(tile->input + p * channels),
          channels * sizeof(float),
          PI_CL_DMA_DIR_EXT2LOC,
          &transfer);
      pi_cl_dma_cmd_wait(&transfer);
    }
  }
}

void pulp_cluster_global_pool_entry(void* context) {
  PulpGlobalPoolArgs* args = (PulpGlobalPoolArgs*)context;
  args->status = -1;
  if (!args->batches || !args->channels) {
    args->status = 0;
    return;
  }
  if (!args->spatial)
    return;
  const size_t overhead = (sizeof(PoolTile) + 7) & ~(size_t)7;
  if (PULP_GLOBAL_POOL_L1_BYTES < overhead + 2 * sizeof(float))
    return;
  const size_t slots = (PULP_GLOBAL_POOL_L1_BYTES - overhead) / sizeof(float);
  // First maximize channels with complete spatial planes. Account for one
  // accumulator per channel and keep every DMA payload below 64 KiB.
  size_t channel_capacity = 0;
  if (args->spatial < slots && args->spatial <= 16380)
    channel_capacity =
        smaller(slots / (args->spatial + 1), 16380 / args->spatial);
  if (!channel_capacity)
    channel_capacity = smaller(NUM_CORES, slots / 2);
  channel_capacity = smaller(channel_capacity, args->channels);
  size_t pixel_capacity = smaller(
      args->spatial,
      smaller(slots / channel_capacity - 1, 16380 / channel_capacity));
  uint8_t* arena = NULL;
  size_t arena_bytes = 0;
  while (channel_capacity && pixel_capacity) {
    arena_bytes =
        overhead + sizeof(float) * channel_capacity * (pixel_capacity + 1);
    arena = (uint8_t*)pi_cl_l1_malloc(0, arena_bytes);
    if (arena)
      break;
    if (channel_capacity > 1)
      channel_capacity /= 2;
    else
      pixel_capacity /= 2;
  }
  if (!arena)
    return;
  PoolTile* tile = (PoolTile*)arena;
  tile->input = (float*)(arena + overhead);
  tile->sums = tile->input + channel_capacity * pixel_capacity;
  tile->inverse_spatial = 1.0f / (float)args->spatial;
  tile->channels_last = args->channels_last;
  for (size_t batch = 0; batch < args->batches; ++batch) {
    for (size_t channel = 0; channel < args->channels;
         channel += tile->channels) {
      tile->channels = smaller(channel_capacity, args->channels - channel);
      for (size_t c = 0; c < tile->channels; ++c)
        tile->sums[c] = 0.0f;
      for (size_t pixel = 0; pixel < args->spatial; pixel += tile->pixels) {
        tile->pixels = smaller(pixel_capacity, args->spatial - pixel);
        tile->last = tile->pixels == args->spatial - pixel;
        load_tile(args, tile, batch, channel, pixel);
        pi_cl_team_fork(NUM_CORES, pooling_worker, tile);
      }
      pi_cl_dma_cmd_t transfer;
      pi_cl_dma_cmd(
          (uintptr_t)(args->output + batch * args->channels + channel),
          (uintptr_t)tile->sums,
          tile->channels * sizeof(float),
          PI_CL_DMA_DIR_LOC2EXT,
          &transfer);
      pi_cl_dma_cmd_wait(&transfer);
    }
  }
  pi_cl_l1_free(0, arena, arena_bytes);
  args->status = 0;
}
