# Copyright (c) 2026 KU Leuven.
# All rights reserved.

cmake_minimum_required(VERSION 3.19)

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)

set(PULP_NN_ROOT
    ""
    CACHE PATH "Path to the PULP-NN source tree"
)
set(PULP_SDK_INCLUDE_DIRS
    ""
    CACHE
      STRING
      "Semicolon-separated PMSIS include directories supplied by the PULP SDK"
)
set(PULP_NUM_CORES
    8
    CACHE STRING "PULP cluster core count"
)

set(PULP_QUANTIZATION_L1_BYTES
    32768
    CACHE STRING "Maximum L1 arena bytes for DMA-tiled quantization"
)
if(PULP_QUANTIZATION_L1_BYTES LESS 256)
  message(FATAL_ERROR "PULP_QUANTIZATION_L1_BYTES must be at least 256")
endif()

set(PULP_ACTIVATION_L1_BYTES
    32768
    CACHE STRING "Maximum L1 arena bytes for DMA-tiled activations"
)
if(PULP_ACTIVATION_L1_BYTES LESS 256)
  message(FATAL_ERROR "PULP_ACTIVATION_L1_BYTES must be at least 256")
endif()

set(PULP_GLOBAL_POOL_L1_BYTES
    32768
    CACHE STRING "Maximum L1 arena bytes for channel-tiled global pooling"
)
if(PULP_GLOBAL_POOL_L1_BYTES LESS 256)
  message(FATAL_ERROR "PULP_GLOBAL_POOL_L1_BYTES must be at least 256")
endif()

option(
  PULP_PROFILE_CONV_TRANSFERS
  "Print standard convolution cluster-stage timings (perturbs FC profiling)"
  OFF
)

if(EXECUTORCH_BUILD_PULP)
  if(NOT EXISTS "${PULP_NN_ROOT}/32bit/include/pulp_nn_kernels.h")
    message(FATAL_ERROR "Set PULP_NN_ROOT to the checked-in PULP-NN tree")
  endif()
  if(NOT PULP_SDK_INCLUDE_DIRS)
    message(FATAL_ERROR "PULP_SDK_INCLUDE_DIRS is required for pmsis.h")
  endif()

  add_library(
    pulp_nn
    ${PULP_NN_ROOT}/32bit/src/pulp_nn_conv_Ho_parallel.c
    ${PULP_NN_ROOT}/32bit/src/pulp_nn_depthwise_generic.c
    ${PULP_NN_ROOT}/32bit/src/pulp_nn_matmul.c
    ${PULP_NN_ROOT}/32bit/src/pulp_nn_pointwise_Co_parallel.c
    ${PULP_NN_ROOT}/32bit/src/pulp_nn_utils.c
  )
  if(NOT EXISTS "${PULP_NN_ROOT}/32bit/include/pulp_nn_bias.h")
    message(FATAL_ERROR "PULP-NN must include the int32-bias extension")
  endif()
  target_compile_definitions(
    pulp_nn PUBLIC NUM_CORES=${PULP_NUM_CORES} PULP_NN_INT32_BIAS
  )
  # Besides being appropriate for PULP-NN, optimization selects the valid
  # inline-function path in this SDK revision's hal/eu/eu_v3.h. Its unoptimized
  # fallback macro contains preprocessor directives inside a continued macro and
  # is rejected by GCC.
  target_compile_options(pulp_nn PRIVATE -O3)
  target_include_directories(pulp_nn PUBLIC ${PULP_NN_ROOT}/32bit/include)
  target_include_directories(pulp_nn PRIVATE ${PULP_SDK_INCLUDE_DIRS})

  set(_pulp_kernel_sources
      ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantized_conv2d.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantized_pointwise.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantized_depthwise_conv2d.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantize_per_tensor.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantize_per_tensor_cluster.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/ops/pulp_activation_cluster.c
      ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_activation_cluster.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/ops/pulp_quantization_cluster.c
      ${CMAKE_CURRENT_SOURCE_DIR}/ops/pulp_global_pool_cluster.c
      ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_global_avg_pool2d.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/ops/pulp_runtime_context.c
  )
  # Legacy conversions and cluster-op wrappers run on the FC without an FPU. The
  # quantization, activation, and pooling C workers execute FP32 on the cluster.
  set_source_files_properties(
    ops/op_global_avg_pool2d.cpp ops/op_activation_cluster.cpp
    ops/op_quantize_per_tensor.cpp ops/op_quantize_per_tensor_cluster.cpp
    PROPERTIES COMPILE_OPTIONS "-march=rv32imc_xpulpv3"
  )
  add_library(pulp_kernels ${_pulp_kernel_sources})
  target_compile_definitions(
    pulp_kernels
    PRIVATE NUM_CORES=${PULP_NUM_CORES}
            PULP_PROFILE_CONV_TRANSFERS=$<BOOL:${PULP_PROFILE_CONV_TRANSFERS}>
            PULP_GLOBAL_POOL_L1_BYTES=${PULP_GLOBAL_POOL_L1_BYTES}
            PULP_ACTIVATION_L1_BYTES=${PULP_ACTIVATION_L1_BYTES}
            PULP_QUANTIZATION_L1_BYTES=${PULP_QUANTIZATION_L1_BYTES}
  )
  # Arithmetic workers require the cluster FP32 ISA; do not compile with
  # fast-math.
  set_source_files_properties(
    ops/pulp_global_pool_cluster.c ops/pulp_quantization_cluster.c
    ops/pulp_activation_cluster.c PROPERTIES COMPILE_OPTIONS
                                             "-ffp-contract=off;-fno-fast-math"
  )
  target_compile_options(pulp_kernels PRIVATE -O3)
  target_include_directories(
    pulp_kernels
    PRIVATE ${EXECUTORCH_ROOT}/..
            ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
            ${PULP_NN_ROOT}/32bit/include
  )
  target_include_directories(
    pulp_kernels PRIVATE $<$<COMPILE_LANGUAGE:C>:${PULP_SDK_INCLUDE_DIRS}>
  )
  target_link_libraries(pulp_kernels PRIVATE pulp_nn executorch_core m)

  set(_yaml_file ${CMAKE_CURRENT_LIST_DIR}/ops/operators.yaml)
  gen_selected_ops(LIB_NAME "pulp_ops_lib" OPS_SCHEMA_YAML "${_yaml_file}")
  generate_bindings_for_kernels(
    LIB_NAME "pulp_ops_lib" CUSTOM_OPS_YAML "${_yaml_file}"
  )
  gen_operators_lib(
    LIB_NAME "pulp_ops_lib" KERNEL_LIBS pulp_kernels DEPS executorch_core
  )

  install(
    TARGETS pulp_nn pulp_kernels pulp_ops_lib
    EXPORT ExecuTorchTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()
