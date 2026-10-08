/*
 * The PULP GCC 9.2 Newlib build provides these C99 functions in the global
 * namespace but configures libstdc++ without _GLIBCXX_USE_C99_MATH_TR1.
 */

#pragma once

#include <math.h>

#ifdef __cplusplus
namespace std {
using ::atanh;
using ::erf;
using ::erfc;
using ::expm1;
using ::lgamma;
using ::log1p;
using ::log2;
using ::nearbyint;
using ::trunc;
} // namespace std
#endif
