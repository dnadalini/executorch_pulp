/* Build-only logging adapter; no solver source modifications. */
#pragma once
#include <stdio.h>

#include "pmsis.h"
int pulp_dynamic_solver_printf(const char* format, ...);
void pulp_dynamic_solver_exit(int code) __attribute__((noreturn));
#define printf pulp_dynamic_solver_printf
#define pmsis_exit pulp_dynamic_solver_exit
