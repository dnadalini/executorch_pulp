/*
 * Copyright (c) 2026 KU Leuven.
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
  uint16_t activation_max_plus_one; // Zero disables the optional upper clamp.
} PulpConv2dArgs;

typedef struct {
  PulpConv2dArgs kernel;
  size_t batches;
} PulpPointwiseArgs;

int pulp_runtime_run_pointwise(PulpPointwiseArgs* args);

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
  int quantize; // bit 0: quantize; bit 1: signed INT8 storage.
  int status;
} PulpQuantizationArgs;

// DMA-tiled conversion: flat physical storage order is preserved.
int pulp_runtime_run_quantization(PulpQuantizationArgs* args);
// Cluster entry, used only by the platform dispatcher.
void pulp_cluster_quantization_entry(void* context);

// ReLU, bounded clamp, exact GELU, or tanh GELU (kind 0, 1, 2, 3).
typedef struct {
  const void* input;
  void* output;
  size_t count;
  float lower, upper, input_scale, output_inverse_scale;
  int32_t input_zero_point, output_zero_point, quant_min, quant_max;
  // Quantized tags: 0=FP32, 1=UINT8, 2=INT8.
  int kind, input_quantized, output_quantized, status;
} PulpActivationArgs;
int pulp_runtime_run_activation(PulpActivationArgs* args);
void pulp_cluster_activation_entry(void* context);

// Channel-tiled FP32 global average pool, logical NCHW / physical NCHW or NHWC.
typedef struct {
  const float* input;
  float* output;
  size_t batches, channels, spatial;
  int channels_last;
  int status;
} PulpGlobalPoolArgs;
int pulp_runtime_run_global_pool(PulpGlobalPoolArgs* args);
void pulp_cluster_global_pool_entry(void* context);

// Same-shaped FP32 addition in matching physical storage order.
typedef struct {
  const float* left;
  const float* right;
  float* output;
  size_t count;
  float alpha;
  int status;
} PulpAddArgs;
int pulp_runtime_run_add(PulpAddArgs* args);
void pulp_cluster_add_entry(void* context);

// Integer affine operation: low two bits 0=requantize, 1=add, 2=pool;
// bit 2 selects signed INT8 storage. Existing UINT8 jobs retain values 0..2.
typedef struct {
  const uint8_t* input;
  const uint8_t* other;
  uint8_t* output;
  size_t count, batches, channels, spatial;
  int64_t ma, mb;
  int shift, za, zb, zo, qmin, qmax;
  int operation, channels_last, status;
} PulpIntegerArgs;
int pulp_runtime_run_integer(PulpIntegerArgs* args);
void pulp_cluster_integer_entry(void* context);

// Platform initialization
int pulp_runtime_initialize(void);
// Synchronous FC submission; callers already on the cluster invoke workers
// directly.
int pulp_runtime_run_task(void (*entry)(void*), void* args);
int pulp_runtime_run_task_with_stack(
    void (*entry)(void*),
    void* args,
    uint32_t master_stack,
    uint32_t slave_stack);
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
