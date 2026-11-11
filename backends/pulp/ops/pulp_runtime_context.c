/*
 * Copyright (c) 2026 ETH Zurich and University of Bologna.
 * All rights reserved.
 */

#include "pulp_runtime_context.h"

#include <string.h>

#include "pmsis.h"
#include "pulp_nn_kernels.h"

#ifndef NUM_CORES
#define NUM_CORES 8
#endif

static struct pi_device cluster_device;
static int cluster_initialized = 0;

/* -------------- */
/* Platform setup */
/* -------------- */


int pulp_runtime_initialize(void) {
  struct pi_cluster_conf config;
  if (cluster_initialized) {
    return 1;
  }
  pi_cluster_conf_init(&config);
  pi_open_from_conf(&cluster_device, &config);
  if (pi_cluster_open(&cluster_device) != 0) {
    return 0;
  }
  cluster_initialized = 1;
  return 1;
}

void pulp_runtime_shutdown(void) {
  if (cluster_initialized) {
    pi_cluster_close(&cluster_device);
    cluster_initialized = 0;
  }
}

int pulp_runtime_is_initialized(void) { return cluster_initialized; }




/* -------------- */
/* Conv2d Workers */
/* -------------- */

static void conv_worker(void* context) {
  PulpConv2dArgs* args = (PulpConv2dArgs*)context;

  // printf("Running Conv2d on Core %d; MemAddr: Input = 0x%x, weights = 0x%x, output = 0x%x\n",
  //     pi_core_id(), (unsigned int)args->input, (unsigned int)args->weight, (unsigned int)args->output);

  pulp_nn_conv_Ho_parallel(
      args->input,
      (uint8_t*)(args + 1),
      args->bias,
      args->output,
      args->weight,
      NULL,
      NULL,
      1,
      args->out_shift,
      args->input_w,
      args->input_h,
      args->input_channels,
      args->output_w,
      args->output_h,
      args->output_channels,
      args->kernel_w,
      args->kernel_h,
      args->padding_top,
      args->padding_bottom,
      args->padding_left,
      args->padding_right,
      args->stride_w,
      args->stride_h,
      0,
      0);
}

static void cluster_conv_entry(void* context) {
  PulpConv2dArgs* fc_args = (PulpConv2dArgs*)context;
  size_t input_bytes =
      (size_t)fc_args->input_h * fc_args->input_w * fc_args->input_channels;
  size_t weight_bytes = (size_t)fc_args->kernel_h * fc_args->kernel_w *
      fc_args->input_channels * fc_args->output_channels;
  size_t output_bytes =
      (size_t)fc_args->output_h * fc_args->output_w * fc_args->output_channels;
  size_t bias_bytes =
      fc_args->bias == NULL ? 0 : sizeof(int32_t) * fc_args->output_channels;
  size_t args_bytes = sizeof(PulpConv2dArgs);
  size_t l1_bytes = args_bytes + fc_args->scratch_bytes + input_bytes +
      bias_bytes + weight_bytes + output_bytes;

  uint8_t* allocation = (uint8_t*)pi_cl_l1_malloc(0, l1_bytes);
  if (allocation == NULL) {
    fc_args->status = -1;
    return;
  }

  PulpConv2dArgs* cluster_args = (PulpConv2dArgs*)allocation;
  *cluster_args = *fc_args;
  uint8_t* cursor = allocation + args_bytes + fc_args->scratch_bytes;
  // Bias precedes the byte tensors so all int32 loads remain aligned.
  cluster_args->bias = NULL;
  if (bias_bytes != 0) {
    cluster_args->bias = (const int32_t*)cursor;
    memcpy(cursor, fc_args->bias, bias_bytes);
    cursor += bias_bytes;
  }
  cluster_args->input = cursor;
  memcpy(cursor, fc_args->input, input_bytes);
  cursor += input_bytes;
  cluster_args->weight = (int8_t*)cursor;
  memcpy(cursor, fc_args->weight, weight_bytes);
  cursor += weight_bytes;
  cluster_args->output = cursor;

  pi_cl_team_fork(NUM_CORES, conv_worker, cluster_args);
  memcpy(fc_args->output, cluster_args->output, output_bytes);
  pi_cl_l1_free(0, allocation, l1_bytes);
  fc_args->status = 0;
}

int pulp_runtime_run_conv2d(PulpConv2dArgs* args) {
  struct pi_cluster_task task;
  if (!cluster_initialized || args == NULL) {
    return 0;
  }
  args->status = -1;
  pi_cluster_send_task_to_cl(
      &cluster_device, pi_cluster_task(&task, cluster_conv_entry, args));
  return args->status == 0;
}



/* ----------------- */
/* Depthwise Workers */
/* ----------------- */

static void depthwise_conv_worker(void* context) {
  PulpConv2dArgs* args = (PulpConv2dArgs*)context;

  // printf("Running Depthwise Conv2d on Core %d; MemAddr: Input = 0x%x, weights = 0x%x, output = 0x%x\n",
  //     pi_core_id(), (unsigned int)args->input, (unsigned int)args->weight, (unsigned int)args->output);

  pulp_nn_depthwise_generic(
      args->input,
      (uint8_t*)(args + 1),
      args->bias,
      args->output,
      args->weight,
      NULL,  // Unused bufferB.
      NULL,
      NULL,
      1,
      args->out_shift,
      args->input_w,
      args->input_h,
      args->input_channels,
      args->output_w,
      args->output_h,
      args->output_channels,
      args->kernel_w,
      args->kernel_h,
      args->padding_top,
      args->padding_bottom,
      args->padding_left,
      args->padding_right,
      args->stride_w,
      args->stride_h,
      0,
      0);
}

static void cluster_depthwise_conv_entry(void* context) {
  PulpConv2dArgs* fc_args = (PulpConv2dArgs*)context;
  // All byte counts are calculated wide before narrowing to the SDK's
  // 32-bit allocation API. The arena keeps each section four-byte aligned.
  const uint64_t input_bytes64 =
      (uint64_t)fc_args->input_h * fc_args->input_w * fc_args->input_channels;
  const uint64_t kernel_size = (uint64_t)fc_args->kernel_h * fc_args->kernel_w;
  const uint64_t weight_bytes64 = kernel_size * fc_args->output_channels;
  const uint64_t output_bytes64 =
      (uint64_t)fc_args->output_h * fc_args->output_w * fc_args->output_channels;
  const uint64_t scratch_bytes64 = NUM_CORES * ((kernel_size + 3) & ~UINT64_C(3));
  const size_t bias_bytes =
      fc_args->bias == NULL ? 0 : sizeof(int32_t) * fc_args->output_channels;
  const uint64_t input_padded = (input_bytes64 + 3) & ~UINT64_C(3);
  const uint64_t weight_padded = (weight_bytes64 + 3) & ~UINT64_C(3);
  const uint64_t total = sizeof(PulpConv2dArgs) + scratch_bytes64 +
      bias_bytes + input_padded + weight_padded + output_bytes64;
  if (total > UINT32_MAX) {
    fc_args->status = -1;
    return;
  }
  const size_t l1_bytes = (size_t)total;
  uint8_t* allocation = (uint8_t*)pi_cl_l1_malloc(0, l1_bytes);
  if (allocation == NULL) {
    fc_args->status = -1;
    return;
  }

  PulpConv2dArgs* cluster_args = (PulpConv2dArgs*)allocation;
  *cluster_args = *fc_args;
  cluster_args->scratch_bytes = (size_t)scratch_bytes64;
  uint8_t* cursor = allocation + sizeof(PulpConv2dArgs) + cluster_args->scratch_bytes;
  cluster_args->bias = NULL;
  if (bias_bytes != 0) {
    cluster_args->bias = (const int32_t*)cursor;
    memcpy(cursor, fc_args->bias, bias_bytes);
    cursor += bias_bytes;
  }

  // External tensors are HWC. The depthwise kernel consumes CHW planes.
  cluster_args->input = cursor;
  const size_t plane_size = (size_t)fc_args->input_h * fc_args->input_w;
  for (size_t c = 0; c < fc_args->input_channels; ++c) {
    for (size_t pixel = 0; pixel < plane_size; ++pixel) {
      cursor[c * plane_size + pixel] =
          fc_args->input[pixel * fc_args->input_channels + c];
    }
  }
  cursor += (size_t)input_padded;
  cluster_args->weight = (const int8_t*)cursor;
  memcpy(cursor, fc_args->weight, (size_t)weight_bytes64);
  cursor += (size_t)weight_padded;
  cluster_args->output = cursor;

  pi_cl_team_fork(NUM_CORES, depthwise_conv_worker, cluster_args);
  memcpy(fc_args->output, cluster_args->output, (size_t)output_bytes64);
  pi_cl_l1_free(0, allocation, l1_bytes);
  fc_args->status = 0;
}

int pulp_runtime_run_depthwise_conv2d(PulpConv2dArgs* args) {
  struct pi_cluster_task task;
  if (!cluster_initialized || args == NULL) {
    return 0;
  }
  args->status = -1;
  pi_cluster_send_task_to_cl(
      &cluster_device, pi_cluster_task(&task, cluster_depthwise_conv_entry, args));
  return args->status == 0;
}


int pulp_runtime_run_quantization(PulpQuantizationArgs* args) {
  struct pi_cluster_task task;
  if (args == NULL) return 0;
  args->status = PULP_QUANTIZATION_RUNTIME_ERROR;
  if (args->count == 0) {
    args->status = 0;
    return 1;
  }
  if (!cluster_initialized) return 0;
  pi_cluster_send_task_to_cl(
      &cluster_device, pi_cluster_task(&task, pulp_cluster_quantization_entry, args));
  return args->status == 0;
}


static void pointwise_worker(void* context) {
  PulpConv2dArgs* args = (PulpConv2dArgs*)context;
  pulp_nn_pointwise_Co_parallel(
      args->input, (uint8_t*)(args + 1), args->bias, args->output, args->weight,
      NULL, NULL, 1, args->out_shift, args->input_w, args->input_h,
      args->input_channels, args->output_w, args->output_h, args->output_channels,
      1, 1, args->padding_top, args->padding_bottom, args->padding_left,
      args->padding_right, args->stride_w, args->stride_h, 0, 0);
}

static void cluster_pointwise_entry(void* context) {
  PulpPointwiseArgs* job = (PulpPointwiseArgs*)context;
  PulpConv2dArgs* args = &job->kernel;
  const uint64_t input_one = (uint64_t)args->input_w * args->input_h * args->input_channels;
  const uint64_t output_one = (uint64_t)args->output_w * args->output_h * args->output_channels;
  const uint64_t input_bytes = input_one * job->batches;
  const uint64_t output_bytes = output_one * job->batches;
  const uint64_t weight_bytes = (uint64_t)args->input_channels * args->output_channels;
  const uint64_t bias_bytes = args->bias == NULL ? 0 : sizeof(int32_t) * args->output_channels;
  const size_t scratch = 2 * NUM_CORES * (((size_t)args->input_channels + 3) & ~(size_t)3);
  const uint64_t input_padded = (input_bytes + 3) & ~UINT64_C(3);
  const uint64_t weight_padded = (weight_bytes + 3) & ~UINT64_C(3);
  const uint64_t total = sizeof(PulpConv2dArgs) + scratch + bias_bytes + input_padded + weight_padded + output_bytes;
  if (total > UINT32_MAX) { args->status = -1; return; }
  uint8_t* arena = (uint8_t*)pi_cl_l1_malloc(0, (size_t)total);
  if (arena == NULL) { args->status = -1; return; }
  PulpConv2dArgs* local = (PulpConv2dArgs*)arena;
  *local = *args;
  local->scratch_bytes = scratch;
  uint8_t* cursor = arena + sizeof(PulpConv2dArgs) + scratch;
  local->bias = NULL;
  if (bias_bytes) {
    local->bias = (const int32_t*)cursor;
    memcpy(cursor, args->bias, (size_t)bias_bytes);
    cursor += (size_t)bias_bytes;
  }
  uint8_t* input = cursor;
  memcpy(input, args->input, (size_t)input_bytes);
  cursor += (size_t)input_padded;
  local->weight = (const int8_t*)cursor;
  memcpy(cursor, args->weight, (size_t)weight_bytes);
  cursor += (size_t)weight_padded;
  uint8_t* output = cursor;
  // Linear vectors use exactly H=W=1, sharing weights and bias across rows.
  for (size_t batch = 0; batch < job->batches; ++batch) {
    local->input = input + batch * (size_t)input_one;
    local->output = output + batch * (size_t)output_one;
    pi_cl_team_fork(NUM_CORES, pointwise_worker, local);
  }
  memcpy(args->output, output, (size_t)output_bytes);
  pi_cl_l1_free(0, arena, (size_t)total);
  args->status = 0;
}

int pulp_runtime_run_pointwise(PulpPointwiseArgs* args) {
  struct pi_cluster_task task;
  if (args == NULL) return 0;
  args->kernel.status = -1;
  if (args->batches == 0) { args->kernel.status = 0; return 1; }
  if (!cluster_initialized) return 0;
  pi_cluster_send_task_to_cl(&cluster_device, pi_cluster_task(&task, cluster_pointwise_entry, args));
  return args->kernel.status == 0;
}
