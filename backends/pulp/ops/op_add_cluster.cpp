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

Tensor& add_cluster_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    const Tensor& other,
    double alpha,
    Tensor& out) {
  if (self.scalar_type() != ScalarType::Float ||
      other.scalar_type() != ScalarType::Float ||
      out.scalar_type() != ScalarType::Float || self.sizes() != other.sizes() ||
      self.dim_order() != other.dim_order() ||
      self.dim_order() != out.dim_order() ||
      self.numel() > std::numeric_limits<size_t>::max() / sizeof(float) ||
      (std::isfinite(alpha) && !std::isfinite(static_cast<float>(alpha)))) {
    ctx.fail(Error::InvalidArgument);
    return out;
  }
  const Error resized = executorch::runtime::resize_tensor(out, self.sizes());
  if (resized != Error::Ok) {
    ctx.fail(resized);
    return out;
  }
  PulpAddArgs args{
      self.const_data_ptr<float>(),
      other.const_data_ptr<float>(),
      out.mutable_data_ptr<float>(),
      static_cast<size_t>(self.numel()),
      static_cast<float>(alpha),
      -1};
  if (!pulp_runtime_run_add(&args))
    ctx.fail(Error::Internal);
  return out;
}
} // namespace pulp::native
