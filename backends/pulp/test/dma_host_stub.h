/* Host-only PMSIS DMA substitutes for the real native-kernel tests. */
#pragma once
#include <assert.h>
#include <stdint.h>
#include <string.h>

typedef struct {
  int unused;
} pi_cl_dma_cmd_t;
#define PI_CL_DMA_DIR_EXT2LOC 1
#define PI_CL_DMA_DIR_LOC2EXT 0

static inline void pi_cl_dma_cmd(
    uintptr_t external,
    uintptr_t local,
    size_t bytes,
    int direction,
    pi_cl_dma_cmd_t* command) {
  (void)command;
  assert(bytes && bytes <= 65535);
  if (direction == PI_CL_DMA_DIR_EXT2LOC)
    memcpy((void*)local, (void*)external, bytes);
  else {
    assert(direction == PI_CL_DMA_DIR_LOC2EXT);
    memcpy((void*)external, (void*)local, bytes);
  }
}

static inline void pi_cl_dma_cmd_2d(
    uintptr_t external,
    uintptr_t local,
    size_t bytes,
    size_t stride,
    size_t length,
    int direction,
    pi_cl_dma_cmd_t* command) {
  (void)command;
  assert(
      bytes && bytes <= 65535 && stride && stride <= 65535 && length &&
      length <= 65535);
  while (bytes) {
    const size_t count = bytes < length ? bytes : length;
    if (direction == PI_CL_DMA_DIR_EXT2LOC)
      memcpy((void*)local, (void*)external, count);
    else {
      assert(direction == PI_CL_DMA_DIR_LOC2EXT);
      memcpy((void*)external, (void*)local, count);
    }
    external += stride;
    local += count;
    bytes -= count;
  }
}
static inline void pi_cl_dma_cmd_wait(pi_cl_dma_cmd_t* command) {
  (void)command;
}
