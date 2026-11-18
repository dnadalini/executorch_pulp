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
bool valid_scale(double scale) {
  return std::isfinite(scale) && scale > 0 &&
      std::isfinite(static_cast<float>(scale)) && static_cast<float>(scale) > 0;
}
} // namespace

Tensor& activation_cluster_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    int64_t kind,
    double min_val,
    double max_val,
    double input_scale,
    int64_t input_zero_point,
    double output_scale,
    int64_t output_zero_point,
    int64_t quant_min,
    int64_t quant_max,
    bool quantized_output,
    Tensor& out) {
  const bool input_quantized = input.scalar_type() == ScalarType::Byte ||
      input.scalar_type() == ScalarType::Char;
  const bool signed_input = input.scalar_type() == ScalarType::Char;
  const bool signed_output = quant_min < 0;
  if ((!input_quantized && input.scalar_type() != ScalarType::Float) ||
      kind < 0 || kind > 3 || std::isnan(min_val) || std::isnan(max_val) ||
      min_val > max_val || input.dim_order() != out.dim_order() ||
      out.scalar_type() !=
          (quantized_output
               ? (signed_output ? ScalarType::Char : ScalarType::Byte)
               : ScalarType::Float) ||
      (input_quantized &&
       (!valid_scale(input_scale) ||
        input_zero_point < (signed_input ? -128 : 0) ||
        input_zero_point > (signed_input ? 127 : 255))) ||
      (quantized_output &&
       (!valid_scale(output_scale) || quant_min < (signed_output ? -128 : 0) ||
        quant_max > (signed_output ? 127 : 255) ||
        quant_min > output_zero_point || output_zero_point > quant_max)) ||
      input.numel() > std::numeric_limits<size_t>::max() / sizeof(float)) {
    ctx.fail(Error::InvalidArgument);
    return out;
  }
  const Error resized = executorch::runtime::resize_tensor(out, input.sizes());
  if (resized != Error::Ok) {
    ctx.fail(resized);
    return out;
  }
  PulpActivationArgs args{
      input.const_data_ptr(),
      out.mutable_data_ptr(),
      static_cast<size_t>(input.numel()),
      static_cast<float>(min_val),
      static_cast<float>(max_val),
      static_cast<float>(input_scale),
      quantized_output ? static_cast<float>(1.0 / output_scale) : 1.0f,
      static_cast<int32_t>(input_zero_point),
      static_cast<int32_t>(output_zero_point),
      static_cast<int32_t>(quant_min),
      static_cast<int32_t>(quant_max),
      static_cast<int>(kind),
      signed_input ? 2 : input_quantized,
      signed_output && quantized_output ? 2 : quantized_output,
      -1};
  if (!pulp_runtime_run_activation(&args)) {
    ctx.fail(
        args.status == PULP_QUANTIZATION_INVALID_INPUT ? Error::InvalidArgument
                                                       : Error::Internal);
  }
  return out;
}
} // namespace pulp::native
