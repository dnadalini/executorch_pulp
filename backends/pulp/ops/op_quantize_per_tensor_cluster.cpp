/* Copyright (c) 2026 KU Leuven. */
#include <executorch/runtime/kernel/kernel_includes.h>

#include <cmath>
#include <limits>
#include "pulp_runtime_context.h"

namespace pulp::native {
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::KernelRuntimeContext;

namespace {
bool prepare(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    double scale,
    int64_t zp,
    int64_t qmin,
    int64_t qmax,
    ScalarType dtype,
    bool quantize,
    Tensor& out) {
  if (!std::isfinite(scale) || scale <= 0 ||
      !std::isfinite(static_cast<float>(scale)) ||
      static_cast<float>(scale) <= 0 || dtype != ScalarType::Byte || qmin < 0 ||
      qmax > 255 || qmin > zp || zp > qmax ||
      input.scalar_type() != (quantize ? ScalarType::Float : dtype) ||
      out.scalar_type() != (quantize ? dtype : ScalarType::Float) ||
      input.dim_order() != out.dim_order()) {
    ctx.fail(Error::InvalidArgument);
    return false;
  }
  const Error err = executorch::runtime::resize_tensor(out, input.sizes());
  if (err != Error::Ok) {
    ctx.fail(err);
    return false;
  }
  return true;
}
} // namespace

Tensor& quantize_per_tensor_cluster_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    double scale,
    int64_t zp,
    int64_t qmin,
    int64_t qmax,
    ScalarType dtype,
    Tensor& out) {
  if (!prepare(ctx, input, scale, zp, qmin, qmax, dtype, true, out))
    return out;
  if (input.numel() > std::numeric_limits<size_t>::max() / sizeof(float)) {
    ctx.fail(Error::InvalidArgument);
    return out;
  }
  PulpQuantizationArgs args{
      input.const_data_ptr<float>(),
      out.mutable_data_ptr<uint8_t>(),
      static_cast<size_t>(input.numel()),
      static_cast<float>(scale),
      static_cast<float>(1.0 / scale),
      static_cast<int32_t>(zp),
      static_cast<int32_t>(qmin),
      static_cast<int32_t>(qmax),
      1,
      -1};
  if (!pulp_runtime_run_quantization(&args)) {
    ctx.fail(
        args.status == PULP_QUANTIZATION_INVALID_INPUT ? Error::InvalidArgument
                                                       : Error::Internal);
  }
  return out;
}

Tensor& dequantize_per_tensor_cluster_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    double scale,
    int64_t zp,
    int64_t qmin,
    int64_t qmax,
    ScalarType dtype,
    Tensor& out) {
  if (!prepare(ctx, input, scale, zp, qmin, qmax, dtype, false, out))
    return out;
  if (input.numel() > std::numeric_limits<size_t>::max() / sizeof(float)) {
    ctx.fail(Error::InvalidArgument);
    return out;
  }
  PulpQuantizationArgs args{
      input.const_data_ptr<uint8_t>(),
      out.mutable_data_ptr<float>(),
      static_cast<size_t>(input.numel()),
      static_cast<float>(scale),
      0.0f,
      static_cast<int32_t>(zp),
      static_cast<int32_t>(qmin),
      static_cast<int32_t>(qmax),
      0,
      -1};
  if (!pulp_runtime_run_quantization(&args))
    ctx.fail(Error::Internal);
  return out;
}
} // namespace pulp::native
