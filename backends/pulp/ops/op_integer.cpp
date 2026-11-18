/* Copyright (c) 2026 KU Leuven. */
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include <limits>

#include "pulp_runtime_context.h"

namespace pulp::native {
using executorch::aten::ArrayRef;
using executorch::aten::DimOrderType;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::KernelRuntimeContext;

namespace {
uint64_t magnitude(int64_t value) {
  return value < 0 ? static_cast<uint64_t>(-(value + 1)) + 1
                   : static_cast<uint64_t>(value);
}
bool parameters(const Tensor& params, PulpIntegerArgs& job, uint64_t bound) {
  const bool add = (job.operation & 3) == 1;
  const int lo = job.operation & 4 ? -128 : 0;
  const int hi = job.operation & 4 ? 127 : 255;
  if (params.scalar_type() != ScalarType::Long || params.dim() != 1 ||
      params.size(0) != (add ? 8 : 6))
    return false;
  const auto* p = params.const_data_ptr<int64_t>();
  const int offset = add ? 2 : 1;
  if (p[offset] < 8 || p[offset] > 46 || p[0] < 0)
    return false;
  for (int i = offset + 1; i < (add ? 8 : 6); ++i)
    if (p[i] < lo || p[i] > hi)
      return false;
  job.ma = p[0];
  job.mb = add ? p[1] : 0;
  const uint64_t maximum = static_cast<uint64_t>(INT64_MAX) / bound;
  if (magnitude(job.ma) > maximum ||
      magnitude(job.mb) > maximum - magnitude(job.ma))
    return false;
  job.shift = static_cast<int>(p[offset]);
  job.za = static_cast<int>(p[offset + 1]);
  job.zb = add ? static_cast<int>(p[offset + 2]) : 0;
  job.zo = static_cast<int>(p[add ? 5 : 3]);
  job.qmin = static_cast<int>(p[add ? 6 : 4]);
  job.qmax = static_cast<int>(p[add ? 7 : 5]);
  return job.qmin <= job.zo && job.zo <= job.qmax;
}

Tensor& elementwise(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor* other,
    const Tensor& params,
    Tensor& out) {
  PulpIntegerArgs job{};
  job.operation =
      (other ? 1 : 0) | (input.scalar_type() == ScalarType::Char ? 4 : 0);
  if ((input.scalar_type() != ScalarType::Byte &&
       input.scalar_type() != ScalarType::Char) ||
      out.scalar_type() != input.scalar_type() ||
      input.dim_order() != out.dim_order() || input.numel() < 0 ||
      static_cast<uint64_t>(input.numel()) >
          std::numeric_limits<size_t>::max() ||
      (other &&
       (other->scalar_type() != input.scalar_type() ||
        other->sizes() != input.sizes() ||
        other->dim_order() != input.dim_order())) ||
      !parameters(params, job, 255)) {
    ctx.fail(Error::InvalidArgument);
    return out;
  }
  const Error resized = executorch::runtime::resize_tensor(out, input.sizes());
  if (resized != Error::Ok) {
    ctx.fail(resized);
    return out;
  }
  job.input = static_cast<const uint8_t*>(input.const_data_ptr());
  job.other =
      other ? static_cast<const uint8_t*>(other->const_data_ptr()) : nullptr;
  job.output = static_cast<uint8_t*>(out.mutable_data_ptr());
  job.count = static_cast<size_t>(input.numel());
  if (!pulp_runtime_run_integer(&job))
    ctx.fail(Error::Internal);
  return out;
}
} // namespace

Tensor& quantized_add_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    const Tensor& other,
    const Tensor& params,
    Tensor& out) {
  return elementwise(ctx, self, &other, params, out);
}
Tensor& requantize_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& params,
    Tensor& out) {
  return elementwise(ctx, input, nullptr, params, out);
}
Tensor& quantized_global_avg_pool2d_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& params,
    bool keepdim,
    Tensor& out) {
  constexpr DimOrderType nchw[] = {0, 1, 2, 3}, nhwc[] = {0, 2, 3, 1};
  const bool channels_last =
      input.dim_order() == ArrayRef<DimOrderType>(nhwc, 4);
  PulpIntegerArgs job{};
  job.operation = 2 | (input.scalar_type() == ScalarType::Char ? 4 : 0);
  if ((input.scalar_type() != ScalarType::Byte &&
       input.scalar_type() != ScalarType::Char) ||
      input.dim() != 4 || out.scalar_type() != input.scalar_type() ||
      out.dim() != (keepdim ? 4 : 2) ||
      (!channels_last &&
       input.dim_order() != ArrayRef<DimOrderType>(nchw, 4)) ||
      input.size(2) <= 0 || input.size(3) <= 0 ||
      (!executorch::runtime::is_contiguous_dim_order(
           out.dim_order().data(), out.dim_order().size()) &&
       !(keepdim && out.dim_order() == ArrayRef<DimOrderType>(nhwc, 4)))) {
    ctx.fail(Error::InvalidArgument);
    return out;
  }
  const uint64_t spatial = static_cast<uint64_t>(input.size(2)) * input.size(3);
  if (spatial > std::numeric_limits<size_t>::max() ||
      spatial > static_cast<uint64_t>(INT64_MAX) / 255 || input.numel() < 0 ||
      static_cast<uint64_t>(input.numel()) >
          std::numeric_limits<size_t>::max() ||
      !parameters(params, job, 255 * spatial)) {
    ctx.fail(Error::InvalidArgument);
    return out;
  }
  Tensor::SizesType sizes[] = {input.size(0), input.size(1), 1, 1};
  const Error resized =
      executorch::runtime::resize_tensor(out, {sizes, keepdim ? 4u : 2u});
  if (resized != Error::Ok) {
    ctx.fail(resized);
    return out;
  }
  job.input = static_cast<const uint8_t*>(input.const_data_ptr());
  job.output = static_cast<uint8_t*>(out.mutable_data_ptr());
  job.count = static_cast<size_t>(input.numel());
  job.batches = input.size(0);
  job.channels = input.size(1);
  job.spatial = spatial;
  job.channels_last = channels_last;
  if (!pulp_runtime_run_integer(&job))
    ctx.fail(Error::Internal);
  return out;
}
} // namespace pulp::native
