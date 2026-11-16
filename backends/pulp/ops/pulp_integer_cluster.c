/* Copyright (c) 2026 KU Leuven. */
#include "pulp_dma.h"
#include "pulp_integer_math.h"
#include "pulp_runtime_context.h"

#ifndef NUM_CORES
#define NUM_CORES 8
#endif
#ifndef PULP_INTEGER_L1_BYTES
#define PULP_INTEGER_L1_BYTES 32768
#endif

typedef struct {
  PulpIntegerArgs job;
  uint8_t* input;
  uint8_t* other;
  int64_t* sums;
  uint8_t* output;
  size_t count, channels, pixels;
  int last;
} IntegerTile;

static size_t minimum(size_t a, size_t b) { return a < b ? a : b; }

static void elementwise_worker(void* context) {
  IntegerTile* tile = (IntegerTile*)context;
  const PulpIntegerArgs* job = &tile->job;
  // Give each core complete four-byte output words, with striped L1 banks.
  for (size_t block = pi_core_id() * 4; block < tile->count;
       block += NUM_CORES * 4) {
    for (size_t i = block; i < block + 4 && i < tile->count; ++i) {
      int64_t value = ((int32_t)tile->input[i] - job->za) * job->ma;
      if (job->operation == 1)
        value += ((int32_t)tile->other[i] - job->zb) * job->mb;
      tile->input[i] =
          pulp_integer_output(pulp_round_shift_even(value, job->shift), job->zo,
                              job->qmin, job->qmax);
    }
  }
}

static void pool_worker(void* context) {
  IntegerTile* tile = (IntegerTile*)context;
  const PulpIntegerArgs* job = &tile->job;
  for (size_t block = pi_core_id() * 4; block < tile->channels;
       block += NUM_CORES * 4) {
    for (size_t c = block; c < block + 4 && c < tile->channels; ++c) {
      int64_t sum = tile->sums[c];
      if (job->channels_last) {
        for (size_t p = 0; p < tile->pixels; ++p)
          sum += (int32_t)tile->input[p * tile->channels + c] - job->za;
      } else {
        for (size_t p = 0; p < tile->pixels; ++p)
          sum += (int32_t)tile->input[c * tile->pixels + p] - job->za;
      }
      tile->sums[c] = sum;
      if (tile->last) {
        int64_t value;
        if (job->spatial == 1)
          value = pulp_round_shift_even(sum * job->ma, job->shift);
        else if (job->spatial > (UINT64_MAX >> job->shift))
          value = 0;
        else
          value = pulp_round_divide_even(sum * job->ma,
                                         (uint64_t)job->spatial << job->shift);
        tile->output[c] =
            pulp_integer_output(value, job->zo, job->qmin, job->qmax);
      }
    }
  }
}

static void load_pool(const PulpIntegerArgs* job, IntegerTile* tile,
                      size_t batch, size_t channel, size_t pixel) {
  const uint8_t* input = job->input + batch * job->channels * job->spatial;
  if (!job->channels_last && tile->pixels == job->spatial) {
    pulp_dma_copy_in(tile->input, input + channel * job->spatial,
                     tile->channels * tile->pixels);
  } else if (!job->channels_last) {
    for (size_t c = 0; c < tile->channels; ++c)
      pulp_dma_copy_in(tile->input + c * tile->pixels,
                       input + (channel + c) * job->spatial + pixel,
                       tile->pixels);
  } else if (tile->channels == job->channels) {
    pulp_dma_copy_in(tile->input, input + pixel * job->channels,
                     tile->channels * tile->pixels);
  } else if (job->channels <= 65535) {
    pi_cl_dma_cmd_t transfer;
    pi_cl_dma_cmd_2d((uintptr_t)(input + pixel * job->channels + channel),
                     (uintptr_t)tile->input, tile->channels * tile->pixels,
                     job->channels, tile->channels, PI_CL_DMA_DIR_EXT2LOC,
                     &transfer);
    pi_cl_dma_cmd_wait(&transfer);
  } else {
    for (size_t p = 0; p < tile->pixels; ++p)
      pulp_dma_copy_in(tile->input + p * tile->channels,
                       input + (pixel + p) * job->channels + channel,
                       tile->channels);
  }
}

static void pool_entry(PulpIntegerArgs* job) {
  const size_t overhead = (sizeof(IntegerTile) + 7) & ~(size_t)7;
  if (PULP_INTEGER_L1_BYTES < overhead + 16) return;
  const size_t available = PULP_INTEGER_L1_BYTES - overhead - 3;
  size_t channels = 0;
  if (job->spatial <= PULP_DMA_MAX_BYTES && job->spatial < available - 9)
    channels = minimum(available / (job->spatial + 9),
                       PULP_DMA_MAX_BYTES / job->spatial);
  if (!channels) channels = minimum(NUM_CORES, available / 10);
  channels = minimum(channels, job->channels);
  size_t pixels = minimum(job->spatial, minimum(available / channels - 9,
                                                PULP_DMA_MAX_BYTES / channels));
  uint8_t* arena = NULL;
  size_t bytes = 0;
  while (channels && pixels) {
    bytes = overhead + channels * 8 + ((channels * pixels + 3) & ~(size_t)3) +
            channels;
    arena = (uint8_t*)pi_cl_l1_malloc(0, bytes);
    if (arena) break;
    if (channels > 1)
      channels /= 2;
    else
      pixels /= 2;
  }
  if (!arena) return;
  IntegerTile* tile = (IntegerTile*)arena;
  tile->job = *job;
  tile->sums = (int64_t*)(arena + overhead);
  tile->input = arena + overhead + channels * 8;
  tile->output = tile->input + ((channels * pixels + 3) & ~(size_t)3);
  for (size_t n = 0; n < job->batches; ++n) {
    for (size_t c = 0; c < job->channels; c += tile->channels) {
      tile->channels = minimum(channels, job->channels - c);
      for (size_t k = 0; k < tile->channels; ++k) tile->sums[k] = 0;
      for (size_t p = 0; p < job->spatial; p += tile->pixels) {
        tile->pixels = minimum(pixels, job->spatial - p);
        tile->last = tile->pixels == job->spatial - p;
        load_pool(job, tile, n, c, p);
        pi_cl_team_fork(NUM_CORES, pool_worker, tile);
      }
      pulp_dma_copy_out(job->output + n * job->channels + c, tile->output,
                        tile->channels);
    }
  }
  pi_cl_l1_free(0, arena, bytes);
  job->status = 0;
}

void pulp_cluster_integer_entry(void* context) {
  PulpIntegerArgs* job = (PulpIntegerArgs*)context;
  job->status = -1;
  if (!job->count) {
    job->status = 0;
    return;
  }
  if (job->operation == 2) {
    pool_entry(job);
    return;
  }
  const size_t overhead = (sizeof(IntegerTile) + 7) & ~(size_t)7;
  const size_t buffers = job->operation == 1 ? 2 : 1;
  if (PULP_INTEGER_L1_BYTES < overhead + buffers * 4) return;
  size_t capacity = minimum((PULP_INTEGER_L1_BYTES - overhead) / buffers,
                            PULP_DMA_MAX_BYTES) &
                    ~(size_t)3;
  if (job->count < capacity) capacity = (job->count + 3) & ~(size_t)3;
  uint8_t* arena = NULL;
  size_t bytes = 0;
  while (capacity >= 4) {
    bytes = overhead + buffers * capacity;
    arena = (uint8_t*)pi_cl_l1_malloc(0, bytes);
    if (arena) break;
    capacity = (capacity / 2) & ~(size_t)3;
  }
  if (!arena) return;
  IntegerTile* tile = (IntegerTile*)arena;
  tile->job = *job;
  tile->input = arena + overhead;
  tile->other = tile->input + capacity;
  for (size_t offset = 0; offset < job->count;) {
    tile->count = minimum(capacity, job->count - offset);
    pulp_dma_copy_in(tile->input, job->input + offset, tile->count);
    if (job->operation == 1)
      pulp_dma_copy_in(tile->other, job->other + offset, tile->count);
    pi_cl_team_fork(NUM_CORES, elementwise_worker, tile);
    pulp_dma_copy_out(job->output + offset, tile->input, tile->count);
    offset += tile->count;
  }
  pi_cl_l1_free(0, arena, bytes);
  job->status = 0;
}
