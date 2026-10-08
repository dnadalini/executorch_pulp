/*
 * Copyright (c) 2026 ETH Zurich and University of Bologna.
 * All rights reserved.
 */

#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include <cstdint>
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

} // namespace

Tensor& quantized_conv2d_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    ArrayRef<int64_t> stride,
    ArrayRef<int64_t> padding,
    int64_t out_shift,
    int64_t scratch_bytes,
    Tensor& out) {
  if (input.dim() != 4 || weight.dim() != 4 || out.dim() != 4 ||
      input.size(0) != 1 || out.size(0) != 1) {
    ET_LOG(
        Error, "pulp::quantized_conv2d expects 4-D tensors and batch size 1");
    context.fail(Error::InvalidArgument);
    return out;
  }
  if (input.scalar_type() != ScalarType::Byte ||
      weight.scalar_type() != ScalarType::Char ||
      out.scalar_type() != ScalarType::Byte) {
    ET_LOG(
        Error,
        "pulp::quantized_conv2d expects uint8 input/output and int8 weight");
    context.fail(Error::InvalidArgument);
    return out;
  }
  if (!is_channels_last(input) || !is_channels_last(out) ||
      !executorch::runtime::is_contiguous_dim_order(
          weight.dim_order().data(), weight.dim_order().size())) {
    ET_LOG(
        Error, "pulp::quantized_conv2d received an incompatible tensor layout");
    context.fail(Error::InvalidArgument);
    return out;
  }
  if (stride.size() != 2 || padding.size() != 2 || out_shift < 0 ||
      out_shift > 31 || scratch_bytes < 0) {
    ET_LOG(
        Error,
        "pulp::quantized_conv2d received invalid convolution parameters");
    context.fail(Error::InvalidArgument);
    return out;
  }

  const int64_t dimensions[] = {
      input.size(3),
      input.size(2),
      input.size(1),
      out.size(3),
      out.size(2),
      out.size(1),
      weight.size(2),
      weight.size(1),
      padding[0],
      padding[0],
      padding[1],
      padding[1],
      stride[1],
      stride[0]};
  for (int64_t dimension : dimensions) {
    if (!fits_u16(dimension)) {
      ET_LOG(Error, "pulp::quantized_conv2d dimension exceeds uint16 range");
      context.fail(Error::InvalidArgument);
      return out;
    }
  }

  const int64_t expected_scratch =
      2 * NUM_CORES * input.size(1) * weight.size(1) * weight.size(2);
  if (scratch_bytes != expected_scratch) {
    ET_LOG(
        Error,
        "pulp::quantized_conv2d scratch mismatch: got %ld expected %ld",
        static_cast<long>(scratch_bytes),
        static_cast<long>(expected_scratch));
    context.fail(Error::InvalidArgument);
    return out;
  }

  PulpConv2dArgs args{input.const_data_ptr<uint8_t>(),
                      weight.const_data_ptr<int8_t>(),
                      out.mutable_data_ptr<uint8_t>(),
                      static_cast<uint16_t>(input.size(3)),
                      static_cast<uint16_t>(input.size(2)),
                      static_cast<uint16_t>(input.size(1)),
                      static_cast<uint16_t>(out.size(3)),
                      static_cast<uint16_t>(out.size(2)),
                      static_cast<uint16_t>(out.size(1)),
                      static_cast<uint16_t>(weight.size(2)),
                      static_cast<uint16_t>(weight.size(1)),
                      static_cast<uint16_t>(padding[0]),
                      static_cast<uint16_t>(padding[0]),
                      static_cast<uint16_t>(padding[1]),
                      static_cast<uint16_t>(padding[1]),
                      static_cast<uint16_t>(stride[1]),
                      static_cast<uint16_t>(stride[0]),
                      static_cast<uint16_t>(out_shift),
                      static_cast<size_t>(scratch_bytes),
                      -1};
  if (!pulp_runtime_run_conv2d(&args)) {
    ET_LOG(Error, "pulp::quantized_conv2d cluster execution failed");
    context.fail(Error::Internal);
  }
  return out;
}

} // namespace pulp::native
