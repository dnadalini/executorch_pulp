/* Copyright (c) 2026 KU Leuven. */
#pragma once

#include <stddef.h>
#include <stdint.h>
#include "pmsis.h"

// MCHAN command lengths are 16-bit. Keep intermediate offsets word-aligned;
// the final command may contain an arbitrary byte tail.
#define PULP_DMA_MAX_BYTES ((size_t)65532)

static inline void pulp_dma_transfer(
    uintptr_t external,
    uintptr_t local,
    size_t bytes,
    int direction) {
  while (bytes) {
    const size_t count =
        bytes > PULP_DMA_MAX_BYTES ? PULP_DMA_MAX_BYTES : bytes;
    pi_cl_dma_cmd_t transfer;
    // The SDK always takes EXT (L2) first, LOC (L1) second, in both directions.
    pi_cl_dma_cmd(external, local, count, direction, &transfer);
    pi_cl_dma_cmd_wait(&transfer);
    external += count;
    local += count;
    bytes -= count;
  }
}

static inline void
pulp_dma_copy_in(void* local, const void* external, size_t bytes) {
  pulp_dma_transfer(
      (uintptr_t)external, (uintptr_t)local, bytes, PI_CL_DMA_DIR_EXT2LOC);
}

static inline void
pulp_dma_copy_out(void* external, const void* local, size_t bytes) {
  pulp_dma_transfer(
      (uintptr_t)external, (uintptr_t)local, bytes, PI_CL_DMA_DIR_LOC2EXT);
}

// Gather one HWC channel directly into a packed CHW plane. The input channel
// count is uint16_t in PulpConv2dArgs and therefore fits the DMA stride field.
static inline void pulp_dma_gather_channel(
    uint8_t* local,
    const uint8_t* external,
    size_t pixels,
    size_t channels) {
  if (channels == 1) {
    pulp_dma_copy_in(local, external, pixels);
    return;
  }
  while (pixels) {
    const size_t count =
        pixels > PULP_DMA_MAX_BYTES ? PULP_DMA_MAX_BYTES : pixels;
    pi_cl_dma_cmd_t transfer;
    pi_cl_dma_cmd_2d(
        (uintptr_t)external,
        (uintptr_t)local,
        count,
        channels,
        1,
        PI_CL_DMA_DIR_EXT2LOC,
        &transfer);
    pi_cl_dma_cmd_wait(&transfer);
    external += count * channels;
    local += count;
    pixels -= count;
  }
}
