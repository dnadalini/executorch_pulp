# Copyright (c) 2026 ETH Zurich and University of Bologna.
# All rights reserved.

cmake_minimum_required(VERSION 3.19)

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)

set(PULP_NN_ROOT "" CACHE PATH "Path to the PULP-NN source tree")
set(PULP_SDK_INCLUDE_DIRS "" CACHE STRING
    "Semicolon-separated PMSIS include directories supplied by the PULP SDK")
set(PULP_NUM_CORES 8 CACHE STRING "PULP cluster core count")

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
    ${PULP_NN_ROOT}/32bit/src/pulp_nn_matmul.c
    ${PULP_NN_ROOT}/32bit/src/pulp_nn_utils.c
  )
  target_compile_definitions(pulp_nn PUBLIC NUM_CORES=${PULP_NUM_CORES})
  target_include_directories(
    pulp_nn PUBLIC ${PULP_NN_ROOT}/32bit/include ${PULP_SDK_INCLUDE_DIRS}
  )

  set(_pulp_kernel_sources
      ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantized_conv2d.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/ops/pulp_runtime_context.c)
  add_library(pulp_kernels ${_pulp_kernel_sources})
  target_compile_definitions(pulp_kernels PRIVATE NUM_CORES=${PULP_NUM_CORES})
  target_include_directories(
    pulp_kernels
    PRIVATE ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
            ${PULP_NN_ROOT}/32bit/include ${PULP_SDK_INCLUDE_DIRS}
  )
  target_link_libraries(
    pulp_kernels PRIVATE pulp_nn executorch_core kernels_util_all_deps
  )

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
