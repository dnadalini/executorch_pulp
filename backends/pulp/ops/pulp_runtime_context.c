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

static void conv_worker(void* context) {
  PulpConv2dArgs* args = (PulpConv2dArgs*)context;
  pulp_nn_conv_Ho_parallel(
      args->input,
      (uint8_t*)(args + 1),
      NULL,
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
  size_t input_bytes = (size_t)fc_args->input_h * fc_args->input_w *
      fc_args->input_channels;
  size_t weight_bytes = (size_t)fc_args->kernel_h * fc_args->kernel_w *
      fc_args->input_channels * fc_args->output_channels;
  size_t output_bytes = (size_t)fc_args->output_h * fc_args->output_w *
      fc_args->output_channels;
  size_t args_bytes = sizeof(PulpConv2dArgs);
  size_t l1_bytes = args_bytes + fc_args->scratch_bytes + input_bytes +
      weight_bytes + output_bytes;

  uint8_t* allocation = (uint8_t*)pi_cl_l1_malloc(0, l1_bytes);
  if (allocation == NULL) {
    fc_args->status = -1;
    return;
  }

  PulpConv2dArgs* cluster_args = (PulpConv2dArgs*)allocation;
  *cluster_args = *fc_args;
  uint8_t* cursor = allocation + args_bytes + fc_args->scratch_bytes;
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

int pulp_runtime_is_initialized(void) {
  return cluster_initialized;
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

