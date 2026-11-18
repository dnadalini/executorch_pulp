/* Copyright (c) 2026 KU Leuven. */
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include <cstdio>
#include <limits>

#include "pulp_dynamic_runtime.h"
namespace pulp_dynamic::native {
using executorch::aten::ArrayRef;
using executorch::aten::DimOrderType;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::KernelRuntimeContext;
Tensor& convolution_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const executorch::aten::optional<Tensor>& bias,
    ArrayRef<int64_t> stride,
    ArrayRef<int64_t> padding,
    int64_t shift,
    int64_t layer,
    const Tensor& config,
    int64_t kind,
    bool linear,
    Tensor& out) {
  auto fail = [&](Error error) -> Tensor& {
    std::printf(
        "PULP_DYNAMIC_FAIL layer=%lld kind=%lld error=%d\n",
        (long long)layer,
        (long long)kind,
        (int)error);
    ET_LOG(
        Error,
        "pulp_dynamic failed layer=%lld kind=%lld",
        (long long)layer,
        (long long)kind);
    ctx.fail(error);
    return out;
  };
  if (input.scalar_type() != ScalarType::Char ||
      weight.scalar_type() != ScalarType::Char ||
      out.scalar_type() != ScalarType::Char ||
      config.scalar_type() != ScalarType::Long || config.dim() != 1 ||
      (config.size(0) != 6 && config.size(0) != 7) ||
      (kind < 0 || kind > 6 || (kind & 3) == 3) || shift < 0 || shift > 31 ||
      layer < 0 || layer > INT32_MAX || stride.size() != 2 ||
      padding.size() != 2)
    return fail(Error::InvalidArgument);
  const auto* p = config.const_data_ptr<int64_t>();
  if (p[0] != PULP_DYNAMIC_L1_BYTES || p[1] < 64 || p[1] > UINT32_MAX ||
      p[2] != pulp_dynamic_double_buffering() || p[3] < 1 ||
      p[3] > UINT32_MAX || (p[4] != 0 && p[4] != 1) || p[5] != NUM_CORES)
    return fail(Error::InvalidArgument);
  PulpDynamicArgs job{};
  job.kind = kind;
  if (config.size(0) == 7) {
    if (!(kind & 4) || p[6] < 1 || p[6] > 128)
      return fail(Error::InvalidArgument);
    job.activation_max_plus_one = p[6];
  }
  kind &= 3;
  job.shift = shift;
  job.layer_index = layer;
  job.l1_bytes = p[0];
  job.metadata_budget = p[1];
  job.max_iterations = p[3];
  job.spatial_mapping_multiple = p[4];
  for (int i = 0; i < 2; ++i)
    if (stride[i] < 1 || stride[i] > 65535 || padding[i] < 0 ||
        padding[i] > 65535)
      return fail(Error::InvalidArgument);
  job.sh = stride[0];
  job.sw = stride[1];
  job.ph = padding[0];
  job.pw = padding[1];
  if ((linear && (kind != 1 || input.dim() < 1)) ||
      (!linear && (input.dim() != 4 || input.size(0) != 1)))
    return fail(Error::InvalidArgument);
  if (weight.dim() != (kind == 0 ? 4 : kind == 1 ? 2 : 3))
    return fail(Error::InvalidArgument);
  job.ci = linear ? input.size(input.dim() - 1) : input.size(1);
  job.co = weight.size(0);
  job.hi = linear ? 1 : input.size(2);
  job.wi = linear ? 1 : input.size(3);
  job.kh = kind == 1 ? 1 : weight.size(1);
  job.kw = kind == 1 ? 1 : weight.size(2);
  for (uint32_t v : {job.ci, job.co, job.hi, job.wi, job.kh, job.kw})
    if (!v || v > 65535)
      return fail(Error::InvalidArgument);
  if ((kind == 0 && weight.size(3) != job.ci) ||
      (kind == 1 &&
       (weight.size(1) != job.ci || job.sh != 1 || job.sw != 1 || job.ph ||
        job.pw)) ||
      (kind == 2 && job.co != job.ci))
    return fail(Error::InvalidArgument);
  constexpr DimOrderType nhwc[] = {0, 2, 3, 1};
  if (linear) {
    if (!executorch::runtime::is_contiguous_dim_order(
            input.dim_order().data(), input.dim_order().size()))
      return fail(Error::InvalidArgument);
  } else if (input.dim_order() != ArrayRef<DimOrderType>(nhwc, 4))
    return fail(Error::InvalidArgument);
  if (out.dim_order() != input.dim_order())
    return fail(Error::InvalidArgument);
  if (job.hi + 2 * job.ph < job.kh || job.wi + 2 * job.pw < job.kw)
    return fail(Error::InvalidArgument);
  job.ho = (job.hi + 2 * job.ph - job.kh) / job.sh + 1;
  job.wo = (job.wi + 2 * job.pw - job.kw) / job.sw + 1;
  if (job.ho > 65535 || job.wo > 65535 || input.numel() < 0 ||
      (uint64_t)input.numel() > UINT32_MAX)
    return fail(Error::InvalidArgument);
  job.rows = linear ? input.numel() / job.ci : 1;
  Tensor::SizesType sizes[16];
  if (input.dim() > 16)
    return fail(Error::InvalidArgument);
  for (int i = 0; i < input.dim(); ++i)
    sizes[i] = input.size(i);
  if (linear)
    sizes[input.dim() - 1] = job.co;
  else {
    sizes[1] = job.co;
    sizes[2] = job.ho;
    sizes[3] = job.wo;
  }
  auto resized =
      executorch::runtime::resize_tensor(out, {sizes, (size_t)input.dim()});
  if (resized != Error::Ok)
    return fail(resized);
  if (bias.has_value() &&
      (bias->scalar_type() != ScalarType::Int || bias->dim() != 1 ||
       bias->size(0) != job.co))
    return fail(Error::InvalidArgument);
  job.input = input.const_data_ptr<int8_t>();
  job.weight = weight.const_data_ptr<int8_t>();
  job.bias = bias.has_value() ? bias->const_data_ptr<int32_t>() : nullptr;
  job.output = out.mutable_data_ptr<int8_t>();
  if (!pulp_dynamic_run(&job))
    return fail(Error::Internal);
  return out;
}
} // namespace pulp_dynamic::native
