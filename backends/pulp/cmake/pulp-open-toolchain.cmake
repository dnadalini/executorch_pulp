# Copyright (c) 2026 ETH Zurich and University of Bologna.
# All rights reserved.

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv32)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

if(NOT DEFINED ENV{PULP_RISCV_GCC_TOOLCHAIN})
  message(FATAL_ERROR "Source setup_pulp_open.sh before configuring CMake")
endif()
if(NOT DEFINED ENV{PULP_SDK_HOME})
  message(FATAL_ERROR "PULP_SDK_HOME is missing; source setup_pulp_open.sh")
endif()

set(_pulp_toolchain "$ENV{PULP_RISCV_GCC_TOOLCHAIN}")
set(_pulp_sdk "$ENV{PULP_SDK_HOME}")
set(_pulp_minimal_libc
    "${_pulp_sdk}/rtos/pulpos/common/lib/libc/minimal/include"
)

set(CMAKE_C_COMPILER "${_pulp_toolchain}/bin/riscv32-unknown-elf-gcc")
set(CMAKE_CXX_COMPILER "${_pulp_toolchain}/bin/riscv32-unknown-elf-g++")
set(CMAKE_AR "${_pulp_toolchain}/bin/riscv32-unknown-elf-ar")
set(CMAKE_RANLIB "${_pulp_toolchain}/bin/riscv32-unknown-elf-ranlib")

set(_pulp_include_dirs
    "${_pulp_sdk}/rtos/pulpos/pulp/include/pos/chips/siracusa"
    "${_pulp_sdk}/rtos/pulpos/pulp/drivers/i3c/include"
    "${_pulp_sdk}/rtos/pulpos/pulp/drivers/siracusa_padmux/include"
    "${_pulp_sdk}/ext_libs/include"
    "${_pulp_minimal_libc}"
    "${_pulp_sdk}/rtos/pulpos/common/include"
    "${_pulp_sdk}/rtos/pulpos/common/kernel"
    "${_pulp_sdk}/rtos/pulpos/pulp_archi/include"
    "${_pulp_sdk}/rtos/pulpos/pulp_hal/include"
    "${_pulp_sdk}/rtos/pmsis/pmsis_api/include"
    "${_pulp_sdk}/rtos/pulpos/pulp/include"
    "${_pulp_sdk}/rtos/pmsis/pmsis_bsp/include"
)

set(PULP_SDK_INCLUDE_DIRS
    "${_pulp_include_dirs}"
    CACHE STRING "PMSIS include directories" FORCE
)

set(_pulp_definitions
    -D__riscv__
    -DCONFIG_SIRACUSA
    -DCONFIG_BOARD_VERSION_SIRACUSA
    -DCONFIG_PROFILE_SIRACUSA
    -D__CONFIG_UDMA__
    -D__PULPOS2__
    -D__PLATFORM__=ARCHI_PLATFORM_GVSOC
    -D__PLATFORM_GVSOC__
    -DARCHI_CLUSTER_NB_PE=8
    -DPOS_CONFIG_IO_UART=0
    -DPOS_CONFIG_IO_UART_BAUDRATE=115200
    -DPOS_CONFIG_IO_UART_ITF=0
)

string(JOIN " " _pulp_definitions_string ${_pulp_definitions})
set(_pulp_common_flags
    "-march=rv32imc_zfinx_xpulpv3 -fdata-sections -ffunction-sections -fno-jump-tables -fno-tree-loop-distribute-patterns -include pos/chips/siracusa/config.h ${_pulp_definitions_string}"
)
set(CMAKE_C_FLAGS_INIT "${_pulp_common_flags}")
set(_pulp_cxx_compat "${CMAKE_CURRENT_LIST_DIR}/pulp-libstdcpp-compat.h")
set(CMAKE_CXX_FLAGS_INIT
    "${_pulp_common_flags} -fno-exceptions -fno-rtti -include ${_pulp_cxx_compat}"
)

foreach(_include_dir IN LISTS _pulp_include_dirs)
  string(APPEND CMAKE_C_FLAGS_INIT " -I${_include_dir}")
  if(NOT _include_dir STREQUAL _pulp_minimal_libc)
    string(APPEND CMAKE_CXX_FLAGS_INIT " -I${_include_dir}")
  endif()
endforeach()

set(EXECUTORCH_PAL_DEFAULT
    minimal
    CACHE STRING "Bare-metal PAL" FORCE
)
set(EXECUTORCH_USE_DL
    OFF
    CACHE BOOL "No dynamic loader on PULP" FORCE
)
