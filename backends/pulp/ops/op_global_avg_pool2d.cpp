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

Tensor& global_avg_pool2d_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    bool keepdim,
    Tensor& out) {
  constexpr DimOrderType nchw[] = {0, 1, 2, 3};
  constexpr DimOrderType nhwc[] = {0, 2, 3, 1};
  const bool channels_last =
      input.dim_order() == ArrayRef<DimOrderType>(nhwc, 4);
  const bool channels_first =
      input.dim_order() == ArrayRef<DimOrderType>(nchw, 4);
  if (input.dim() != 4 || input.scalar_type() != ScalarType::Float ||
      out.scalar_type() != ScalarType::Float ||
      out.dim() != (keepdim ? 4 : 2) || (!channels_first && !channels_last) ||
      input.size(2) <= 0 || input.size(3) <= 0 ||
      (!executorch::runtime::is_contiguous_dim_order(
           out.dim_order().data(), out.dim_order().size()) &&
       !(keepdim && out.dim_order() == ArrayRef<DimOrderType>(nhwc, 4)))) {
    ctx.fail(Error::InvalidArgument);
    return out;
  }
  size_t elements = 1;
  for (size_t d = 0; d < 4; ++d) {
    if (input.size(d) < 0 ||
        static_cast<uint64_t>(input.size(d)) >
            std::numeric_limits<size_t>::max() / sizeof(float) / elements) {
      ctx.fail(Error::InvalidArgument);
      return out;
    }
    // Zero batch/channel is a valid empty output. Keep the overflow divisor
    // nonzero.
    elements *= input.size(d) ? static_cast<size_t>(input.size(d)) : 1;
  }
  Tensor::SizesType sizes[] = {input.size(0), input.size(1), 1, 1};
  const Error resized =
      executorch::runtime::resize_tensor(out, {sizes, keepdim ? 4u : 2u});
  if (resized != Error::Ok) {
    ctx.fail(resized);
    return out;
  }
  PulpGlobalPoolArgs args{
      input.const_data_ptr<float>(),
      out.mutable_data_ptr<float>(),
      static_cast<size_t>(input.size(0)),
      static_cast<size_t>(input.size(1)),
      static_cast<size_t>(input.size(2)) * static_cast<size_t>(input.size(3)),
      channels_last,
      -1};
  if (!pulp_runtime_run_global_pool(&args))
    ctx.fail(Error::Internal);
  return out;
}
} // namespace pulp::native
