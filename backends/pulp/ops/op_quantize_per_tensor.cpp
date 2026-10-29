/* Copyright (c) 2026 ETH Zurich and University of Bologna. */
#include <executorch/runtime/kernel/kernel_includes.h>

#include <cmath>

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

Tensor& quantize_per_tensor_out(
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
  const float inverse_scale = static_cast<float>(1.0 / scale);
  const float* src = input.const_data_ptr<float>();
  uint8_t* dst = out.mutable_data_ptr<uint8_t>();

  printf(
      "[quantize_per_tensor_out] input=%p output=%p scale=%g zp=%ld qmin=%ld "
      "qmax=%ld\n",
      static_cast<const void*>(src),
      static_cast<void*>(dst),
      scale,
      zp,
      qmin,
      qmax);

  for (size_t i = 0; i < input.numel(); ++i) {
    const float value = src[i] * inverse_scale;
    if (std::isnan(value)) {
      ctx.fail(Error::InvalidArgument);
      return out;
    }
    // Saturate before converting to integer, including +/- infinity.
    if (value <= qmin - zp) {
      dst[i] = qmin;
      continue;
    }
    if (value >= qmax - zp) {
      dst[i] = qmax;
      continue;
    }
    // Round to nearest even, independent of the target's floating-point mode.
    int32_t rounded = static_cast<int32_t>(value);
    if (static_cast<float>(rounded) > value)
      --rounded;
    const float lower = static_cast<float>(rounded);
    const float fraction = value - lower;
    if (fraction > 0.5f || (fraction == 0.5f && (rounded & 1)))
      ++rounded;
    dst[i] = static_cast<uint8_t>(rounded + zp);
  }
  return out;
}

Tensor& dequantize_per_tensor_out(
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
  const uint8_t* src = input.const_data_ptr<uint8_t>();
  float* dst = out.mutable_data_ptr<float>();

  printf(
      "[dequantize_per_tensor_out] input=%p output=%p scale=%g zp=%ld qmin=%ld "
      "qmax=%ld\n",
      static_cast<const void*>(src),
      static_cast<void*>(dst),
      scale,
      zp,
      qmin,
      qmax);

  for (size_t i = 0; i < input.numel(); ++i) {
    dst[i] = (static_cast<int32_t>(src[i]) - static_cast<int32_t>(zp)) *
        static_cast<float>(scale);
  }
  return out;
}
} // namespace pulp::native
