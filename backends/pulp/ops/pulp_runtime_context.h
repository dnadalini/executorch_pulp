/*
 * Copyright (c) 2026 ETH Zurich and University of Bologna.
 * All rights reserved.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const uint8_t* input;
  const int8_t* weight;
  uint8_t* output;
  uint16_t input_w;
  uint16_t input_h;
  uint16_t input_channels;
  uint16_t output_w;
  uint16_t output_h;
  uint16_t output_channels;
  uint16_t kernel_w;
  uint16_t kernel_h;
  uint16_t padding_top;
  uint16_t padding_bottom;
  uint16_t padding_left;
  uint16_t padding_right;
  uint16_t stride_w;
  uint16_t stride_h;
  uint16_t out_shift;
  size_t scratch_bytes;
  int status;
} PulpConv2dArgs;

int pulp_runtime_initialize(void);
void pulp_runtime_shutdown(void);
int pulp_runtime_is_initialized(void);
int pulp_runtime_run_conv2d(PulpConv2dArgs* args);

#ifdef __cplusplus
}
#endif
