# Copyright (c) 2026 KU Leuven.
set(PULP_DYNAMIC_SOLVER_ROOT
    "${EXECUTORCH_ROOT}/../on-device-solver"
    CACHE PATH "On-device solver checkout"
)
set(PULP_DYNAMIC_L1_BYTES
    212000
    CACHE STRING "Tile-buffer budget in bytes"
)
option(PULP_DYNAMIC_DOUBLE_BUFFERING
       "Compile solver and templates for double buffering" ON
)
option(PULP_DYNAMIC_VERBOSE "Print solver diagnostics (perturbs latency)" OFF)
if(NOT EXISTS "${PULP_DYNAMIC_SOLVER_ROOT}/lib/inc/solver.h")
  message(FATAL_ERROR "Initialize the on-device-solver submodule")
endif()
if(PULP_DYNAMIC_L1_BYTES LESS 256)
  message(FATAL_ERROR "Invalid PULP_DYNAMIC_L1_BYTES")
endif()
set(_solver_src "${PULP_DYNAMIC_SOLVER_ROOT}/lib/src")
set(_solver_inc "${PULP_DYNAMIC_SOLVER_ROOT}/lib/inc")
add_library(
  pulp_dynamic_solver
  ${_solver_src}/solver.c
  ${_solver_src}/templates_int8.c
  ${_solver_src}/templates_fp32.c
  ${_solver_src}/pulp_cnn_kernels.c
  ${_solver_src}/pulp_utils.c
  ${_solver_src}/tiling_builder_loop_orders.c
  ${_solver_src}/saturation/saturation.c
  ${_solver_src}/saturation/saturation_solver.c
  ${_solver_src}/pulp_nn_kernels/pulp_nn_conv.c
  ${_solver_src}/pulp_nn_kernels/pulp_nn_matmul.c
)
math(EXPR _l1_kb "${PULP_DYNAMIC_L1_BYTES} / 1000")
# Match the tile budget, leaving no implicit L1 metadata allowance. Metadata
# then uses L2, while the remaining physical L1 is reserved for runtime/stacks.
target_compile_definitions(
  pulp_dynamic_solver
  PRIVATE NUM_CORES=${PULP_NUM_CORES}
          USE_DOUBLE_BUFFERING=$<BOOL:${PULP_DYNAMIC_DOUBLE_BUFFERING}>
          FC_FREQUENCY_MHZ=370
          CL_FREQUENCY_MHZ=370
          L1_SIZE_KB=${_l1_kb}
          HWC_LAYOUT=1
          USE_IM2COL=1
          NO_PULP_TRAINLIB_IM2COL
)
target_include_directories(
  pulp_dynamic_solver
  PRIVATE ${PULP_SDK_INCLUDE_DIRS} ${_solver_inc} ${_solver_inc}/saturation
          ${_solver_inc}/pulp_nn_kernels
)
target_compile_options(
  pulp_dynamic_solver
  PRIVATE -O3 -ffp-contract=off -fno-fast-math -include
          ${CMAKE_CURRENT_SOURCE_DIR}/ops/solver_build_shim.h
)
target_link_libraries(pulp_dynamic_solver PRIVATE m)
add_library(
  pulp_dynamic_kernels ops/op_convolution.cpp ops/pulp_dynamic_runtime.c
)
target_include_directories(
  pulp_dynamic_kernels
  PRIVATE ${EXECUTORCH_ROOT}/..
          ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
          ${EXECUTORCH_ROOT}/backends/pulp/ops
          ${_solver_inc}
          $<$<COMPILE_LANGUAGE:C>:${PULP_SDK_INCLUDE_DIRS}>
)
target_compile_definitions(
  pulp_dynamic_kernels
  PRIVATE NUM_CORES=${PULP_NUM_CORES}
          PULP_DYNAMIC_L1_BYTES=${PULP_DYNAMIC_L1_BYTES}
          USE_DOUBLE_BUFFERING=$<BOOL:${PULP_DYNAMIC_DOUBLE_BUFFERING}>
          PULP_DYNAMIC_VERBOSE=$<BOOL:${PULP_DYNAMIC_VERBOSE}>
)
target_compile_options(pulp_dynamic_kernels PRIVATE -O3)
set_source_files_properties(
  ops/op_convolution.cpp PROPERTIES COMPILE_OPTIONS "-march=rv32imc_xpulpv3"
)
target_link_libraries(
  pulp_dynamic_kernels
  PUBLIC pulp_runtime_common
  PRIVATE pulp_dynamic_solver executorch_core m
)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)
gen_selected_ops(
  LIB_NAME pulp_dynamic_ops_lib OPS_SCHEMA_YAML
  "${CMAKE_CURRENT_SOURCE_DIR}/ops/operators.yaml"
)
generate_bindings_for_kernels(
  LIB_NAME pulp_dynamic_ops_lib CUSTOM_OPS_YAML
  "${CMAKE_CURRENT_SOURCE_DIR}/ops/operators.yaml"
)
gen_operators_lib(
  LIB_NAME pulp_dynamic_ops_lib KERNEL_LIBS pulp_dynamic_kernels DEPS
  executorch_core
)
install(
  TARGETS pulp_dynamic_solver pulp_dynamic_kernels pulp_dynamic_ops_lib
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
message(
  STATUS
    "pulp_dynamic: L1=${PULP_DYNAMIC_L1_BYTES} bytes, double_buffering=${PULP_DYNAMIC_DOUBLE_BUFFERING}, verbose=${PULP_DYNAMIC_VERBOSE}"
)
