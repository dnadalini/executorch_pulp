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
  const int32_t* bias;
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

enum {
  PULP_QUANTIZATION_RUNTIME_ERROR = -1,
  PULP_QUANTIZATION_INVALID_INPUT = -2,
};

typedef struct {
  const void* input;
  void* output;
  size_t count;
  float scale;
  float inverse_scale;
  int32_t zero_point;
  int32_t quant_min;
  int32_t quant_max;
  int quantize;
  int status;
} PulpQuantizationArgs;

// DMA-tiled conversion: flat physical storage order is preserved.
int pulp_runtime_run_quantization(PulpQuantizationArgs* args);
// Cluster entry, used only by the platform dispatcher.
void pulp_cluster_quantization_entry(void* context);

// Platform initialization
int pulp_runtime_initialize(void);
void pulp_runtime_shutdown(void);
int pulp_runtime_is_initialized(void);

// Convolution workers
int pulp_runtime_run_conv2d(PulpConv2dArgs* args);
// Multiplier-1 depthwise: external HWC activations, OIHW weights.
// Converts input to CHW and derives scratch_bytes internally.
int pulp_runtime_run_depthwise_conv2d(PulpConv2dArgs* args);

#ifdef __cplusplus
}
#endif
