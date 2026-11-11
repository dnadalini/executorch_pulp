/*
 * Copyright (c) 2026 ETH Zurich and University of Bologna.
 * All rights reserved.
 */

#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include <cstdint>
#include <cstring>
#include <limits>

#include "pulp_runtime_context.h"

#ifndef NUM_CORES
#define NUM_CORES 8
#endif

namespace pulp::native {

using executorch::aten::ArrayRef;
using executorch::aten::ScalarType;
using executorch::runtime::Error;
using torch::executor::KernelRuntimeContext;
using torch::executor::Tensor;

namespace {

bool is_channels_last(const Tensor& tensor) {
  constexpr executorch::aten::DimOrderType order[] = {0, 2, 3, 1};
  return tensor.dim() == 4 &&
      tensor.dim_order() == ArrayRef<executorch::aten::DimOrderType>(order, 4);
}

bool fits_u16(int64_t value) {
  return value >= 0 && value <= std::numeric_limits<uint16_t>::max();
}

bool requantization_shift(const Tensor& scales, int& shift) {
  // Interpret positive IEEE float32 scales as mantissa * 2**exponent. The FC
  // must not require floating-point instructions to configure a cluster op.
  uint32_t mantissa[3];
  int exponent[3];
  const auto* bytes = static_cast<const uint8_t*>(scales.const_data_ptr());
  for (size_t i = 0; i < 3; ++i) {
    uint32_t bits;
    std::memcpy(&bits, bytes + i * sizeof(bits), sizeof(bits));
    const uint32_t encoded_exponent = (bits >> 23) & 0xff;
    mantissa[i] = bits & 0x7fffff;
    if ((bits >> 31) || encoded_exponent == 0xff ||
        (encoded_exponent == 0 && mantissa[i] == 0)) {
      return false;
    }
    exponent[i] = encoded_exponent == 0
        ? -149
        : static_cast<int>(encoded_exponent) - 127 - 23;
    if (encoded_exponent != 0) {
      mantissa[i] |= 1u << 23;
    }
    while (mantissa[i] < (1u << 23)) {
      mantissa[i] <<= 1;
      --exponent[i];
    }
  }
  const uint64_t product = static_cast<uint64_t>(mantissa[0]) * mantissa[1];
  for (int candidate = 0; candidate <= 31; ++candidate) {
    const int power = exponent[2] - exponent[0] - exponent[1] - candidate;
    // Normalized mantissas constrain this alignment to at most 25 bits.
    if (power < 22 || power > 25) {
      continue;
    }
    const uint64_t expected = static_cast<uint64_t>(mantissa[2]) << power;
    const uint64_t difference =
        product > expected ? product - expected : expected - product;
    if (difference <= expected / 100000) {
      shift = candidate;
      return true;
    }
  }
  return false;
}

} // namespace

Tensor& quantized_pointwise_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const executorch::aten::optional<Tensor>& bias,
    const Tensor& scales,
    const Tensor& zero_points,
    ArrayRef<int64_t> stride,
    ArrayRef<int64_t> padding,
    bool linear,
    Tensor& out) {
  auto invalid = [&]() { context.fail(Error::InvalidArgument); };
  if (weight.dim() != 2 || input.dim() < 1 || input.dim() > 16 ||
      out.dim() != input.dim() || input.scalar_type() != ScalarType::Byte ||
      weight.scalar_type() != ScalarType::Char ||
      out.scalar_type() != ScalarType::Byte ||
      !executorch::runtime::is_contiguous_dim_order(
          weight.dim_order().data(), weight.dim_order().size()) ||
      stride.size() != 2 || padding.size() != 2) {
    invalid();
    return out;
  }
  if (linear) {
    if (!executorch::runtime::is_contiguous_dim_order(
            input.dim_order().data(), input.dim_order().size()) ||
        !executorch::runtime::is_contiguous_dim_order(
            out.dim_order().data(), out.dim_order().size()) ||
        stride[0] != 1 || stride[1] != 1 || padding[0] || padding[1]) {
      invalid();
      return out;
    }
  } else if (
      input.dim() != 4 || input.size(0) != 1 || !is_channels_last(input) ||
      !is_channels_last(out)) {
    invalid();
    return out;
  }
  const int64_t channels = input.size(linear ? input.dim() - 1 : 1);
  if (channels <= 0 || !fits_u16(channels) || weight.size(1) != channels ||
      weight.size(0) <= 0 || !fits_u16(weight.size(0)) || stride[0] <= 0 ||
      stride[1] <= 0 || !fits_u16(stride[0]) || !fits_u16(stride[1]) ||
      !fits_u16(padding[0]) || !fits_u16(padding[1])) {
    invalid();
    return out;
  }
  if (scales.scalar_type() != ScalarType::Float || scales.dim() != 1 ||
      scales.size(0) != 3 || zero_points.scalar_type() != ScalarType::Int ||
      zero_points.dim() != 1 || zero_points.size(0) != 3 ||
      (bias.has_value() &&
       (bias->scalar_type() != ScalarType::Int || bias->dim() != 1 ||
        bias->size(0) != weight.size(0)))) {
    invalid();
    return out;
  }
  for (size_t i = 0; i < 3; ++i)
    if (zero_points.const_data_ptr<int32_t>()[i] != 0) {
      invalid();
      return out;
    }
  int shift;
  if (!requantization_shift(scales, shift)) {
    invalid();
    return out;
  }
  Tensor::SizesType sizes[16];
  for (size_t i = 0; i < input.dim(); ++i)
    sizes[i] = input.size(i);
  int64_t input_h = 1, input_w = 1, output_h = 1, output_w = 1;
  size_t batches = 1;
  if (linear) {
    sizes[input.dim() - 1] = weight.size(0);
    const uint64_t rows = static_cast<uint64_t>(input.numel()) / channels;
    if (rows > UINT32_MAX) {
      invalid();
      return out;
    }
    batches = static_cast<size_t>(rows);
  } else {
    input_h = input.size(2);
    input_w = input.size(3);
    if (input_h <= 0 || input_w <= 0 || !fits_u16(input_h) ||
        !fits_u16(input_w)) {
      invalid();
      return out;
    }
    output_h = (input_h + 2 * padding[0] - 1) / stride[0] + 1;
    output_w = (input_w + 2 * padding[1] - 1) / stride[1] + 1;
    if (!fits_u16(output_h) || !fits_u16(output_w)) {
      invalid();
      return out;
    }
    sizes[1] = weight.size(0);
    sizes[2] = output_h;
    sizes[3] = output_w;
  }
  const Error resized = executorch::runtime::resize_tensor(
      out, {sizes, static_cast<size_t>(input.dim())});
  if (resized != Error::Ok) {
    context.fail(resized);
    return out;
  }
  PulpPointwiseArgs args{};
  args.kernel = PulpConv2dArgs{
      input.const_data_ptr<uint8_t>(),
      weight.const_data_ptr<int8_t>(),
      bias.has_value() ? bias->const_data_ptr<int32_t>() : nullptr,
      out.mutable_data_ptr<uint8_t>(),
      static_cast<uint16_t>(input_w),
      static_cast<uint16_t>(input_h),
      static_cast<uint16_t>(channels),
      static_cast<uint16_t>(output_w),
      static_cast<uint16_t>(output_h),
      static_cast<uint16_t>(weight.size(0)),
      1,
      1,
      static_cast<uint16_t>(padding[0]),
      static_cast<uint16_t>(padding[0]),
      static_cast<uint16_t>(padding[1]),
      static_cast<uint16_t>(padding[1]),
      static_cast<uint16_t>(stride[1]),
      static_cast<uint16_t>(stride[0]),
      static_cast<uint16_t>(shift),
      0,
      -1};
  args.batches = batches;
  if (!pulp_runtime_run_pointwise(&args)) {
    context.fail(Error::Internal);
  }
  return out;
}
} // namespace pulp::native
