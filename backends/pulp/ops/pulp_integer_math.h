/* Copyright (c) 2026 KU Leuven. */
#pragma once
#include <stdint.h>

// The validated accumulator excludes INT64_MIN; use unsigned magnitude so
// shifts and remainder arithmetic have defined behavior for negative values.
static inline int64_t pulp_round_divide_even(
    int64_t value,
    uint64_t denominator) {
  const uint64_t magnitude =
      value < 0 ? (uint64_t)(-(value + 1)) + 1 : (uint64_t)value;
  uint64_t whole = magnitude / denominator;
  const uint64_t remainder = magnitude % denominator;
  if (remainder > denominator / 2 ||
      (!(denominator & 1) && remainder == denominator / 2 && (whole & 1)))
    ++whole;
  return value < 0 ? -(int64_t)whole : (int64_t)whole;
}

static inline int64_t pulp_round_shift_even(int64_t value, unsigned shift) {
  const uint64_t magnitude =
      value < 0 ? (uint64_t)(-(value + 1)) + 1 : (uint64_t)value;
  uint64_t whole = magnitude >> shift;
  const uint64_t remainder = magnitude & ((UINT64_C(1) << shift) - 1);
  const uint64_t half = UINT64_C(1) << (shift - 1);
  if (remainder > half || (remainder == half && (whole & 1)))
    ++whole;
  return value < 0 ? -(int64_t)whole : (int64_t)whole;
}

static inline uint8_t
pulp_integer_output(int64_t value, int zp, int qmin, int qmax) {
  value += zp;
  return value < qmin ? (uint8_t)qmin
      : value > qmax  ? (uint8_t)qmax
                      : (uint8_t)value;
}
