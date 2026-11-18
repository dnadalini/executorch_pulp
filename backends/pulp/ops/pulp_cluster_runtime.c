/* Copyright (c) 2026 KU Leuven. */
#include "pmsis.h"
#include "pulp_runtime_context.h"

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

int pulp_runtime_run_task_with_stack(void (*entry)(void*), void* args,
                                     uint32_t master_stack,
                                     uint32_t slave_stack) {
  if (!cluster_initialized || entry == NULL) return 0;
  struct pi_cluster_task task;
  pi_cluster_task(&task, entry, args);
#ifdef __PULPOS2__
  task.stack_size = master_stack;
  task.slave_stack_size = slave_stack;
#else
  (void)master_stack;
  (void)slave_stack;
#endif
  pi_cluster_send_task_to_cl(&cluster_device, &task);
  return 1;
}

int pulp_runtime_run_task(void (*entry)(void*), void* args) {
  return pulp_runtime_run_task_with_stack(entry, args, 0, 0);
}

int pulp_runtime_run_quantization(PulpQuantizationArgs* args) {
  if (args == NULL) return 0;
  args->status = PULP_QUANTIZATION_RUNTIME_ERROR;
  if (args->count == 0) {
    args->status = 0;
    return 1;
  }
  if (!pulp_runtime_is_initialized()) return 0;
  pulp_runtime_run_task(pulp_cluster_quantization_entry, args);
  return args->status == 0;
}

int pulp_runtime_run_activation(PulpActivationArgs* args) {
  if (args == NULL) return 0;
  args->status = PULP_QUANTIZATION_RUNTIME_ERROR;
  if (args->count == 0) {
    args->status = 0;
    return 1;
  }
  if (!pulp_runtime_is_initialized()) return 0;
  pulp_runtime_run_task(pulp_cluster_activation_entry, args);
  return args->status == 0;
}

int pulp_runtime_run_global_pool(PulpGlobalPoolArgs* args) {
  if (args == NULL) return 0;
  args->status = -1;
  if (!args->batches || !args->channels) {
    args->status = 0;
    return 1;
  }
  if (!pulp_runtime_is_initialized() || !args->spatial) return 0;
  pulp_runtime_run_task(pulp_cluster_global_pool_entry, args);
  return args->status == 0;
}

int pulp_runtime_run_add(PulpAddArgs* args) {
  if (args == NULL) return 0;
  args->status = -1;
  if (!args->count) {
    args->status = 0;
    return 1;
  }
  if (!pulp_runtime_is_initialized()) return 0;
  pulp_runtime_run_task(pulp_cluster_add_entry, args);
  return args->status == 0;
}

int pulp_runtime_run_integer(PulpIntegerArgs* args) {
  if (args == NULL) return 0;
  args->status = -1;
  if (!args->count) {
    args->status = 0;
    return 1;
  }
  if (!pulp_runtime_is_initialized()) return 0;
  pulp_runtime_run_task(pulp_cluster_integer_entry, args);
  return args->status == 0;
}
