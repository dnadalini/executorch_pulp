# Copyright (c) 2026 ETH Zurich and University of Bologna.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch.library import impl, Library, register_fake


lib = Library("pulp", "DEF")

lib.define(
    "quantized_conv2d(Tensor input, Tensor weight, Tensor? bias, Tensor scales, "
    "Tensor zero_points, int[] stride, int[] padding, int scratch_bytes) -> Tensor"
)
lib.define(
    "quantized_conv2d.out(Tensor input, Tensor weight, Tensor? bias, Tensor scales, "
    "Tensor zero_points, int[] stride, int[] padding, int scratch_bytes, *, Tensor(a!) out) -> Tensor(a!)"
)


def _output_shape(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride: Sequence[int],
    padding: Sequence[int],
) -> tuple[int, int, int, int]:
    batch, _, input_h, input_w = input.shape
    output_channels, kernel_h, kernel_w, _ = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    output_h = (input_h + 2 * pad_h - kernel_h) // stride_h + 1
    output_w = (input_w + 2 * pad_w - kernel_w) // stride_w + 1
    return batch, output_channels, output_h, output_w


@register_fake("pulp::quantized_conv2d")
def quantized_conv2d_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    stride: Sequence[int],
    padding: Sequence[int],
    scratch_bytes: int,
) -> torch.Tensor:
    del bias, scales, zero_points, scratch_bytes
    return torch.empty(
        _output_shape(input, weight, stride, padding),
        dtype=torch.uint8,
        device=input.device,
        memory_format=torch.channels_last,
    )


@impl(lib, "quantized_conv2d", "CompositeExplicitAutograd")
def quantized_conv2d_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    stride: Sequence[int],
    padding: Sequence[int],
    scratch_bytes: int,
) -> torch.Tensor:
    del scratch_bytes
    if input.dtype != torch.uint8 or weight.dtype != torch.int8:
        raise ValueError("PULP-NN convolution expects uint8 input and int8 weight")
    if input.dim() != 4 or weight.dim() != 4 or input.shape[0] != 1:
        raise ValueError("PULP-NN convolution expects 4-D tensors and batch size 1")
    if (
        scales.dtype != torch.float32
        or scales.shape != (3,)
        or not torch.isfinite(scales).all()
        or (scales <= 0).any()
    ):
        raise ValueError(
            "PULP-NN scales must be three finite positive float32 values: input, weight, output"
        )
    if (
        zero_points.dtype != torch.int32
        or zero_points.shape != (3,)
        or zero_points.any()
    ):
        raise ValueError("PULP-NN requires three int32 zero points equal to zero")
    ratio = float(scales[0]) * float(scales[1]) / float(scales[2])
    out_shift = round(-math.log2(ratio))
    if not 0 <= out_shift <= 31 or not math.isclose(
        ratio, 2.0**-out_shift, rel_tol=1e-5
    ):
        raise ValueError(
            "PULP-NN requires power-of-two requantization with shift in [0, 31]"
        )
    if bias is not None and (
        bias.dtype != torch.int32 or bias.shape != (weight.shape[0],)
    ):
        raise ValueError("PULP-NN bias must be int32 with one value per output channel")

    weight_oihw = weight.permute(0, 3, 1, 2).contiguous()
    accumulator = F.conv2d(
        # float64 represents every int32 dot-product result exactly; float32
        # loses low bits before the final right shift for large accumulators.
        input.to(torch.float64),
        weight_oihw.to(torch.float64),
        bias=None,
        stride=tuple(stride),
        padding=tuple(padding),
    ).to(torch.int64)
    if bias is not None:
        accumulator += bias.to(torch.int64).reshape(1, -1, 1, 1)
    output = torch.bitwise_right_shift(accumulator, out_shift).clamp(0, 255)
    return output.to(torch.uint8, memory_format=torch.channels_last)


@impl(lib, "quantized_conv2d.out", "CompositeExplicitAutograd")
def quantized_conv2d_out_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    stride: Sequence[int],
    padding: Sequence[int],
    scratch_bytes: int,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    result = quantized_conv2d_impl(
        input, weight, bias, scales, zero_points, stride, padding, scratch_bytes
    )
    out.resize_(result.shape, memory_format=torch.channels_last)
    out.copy_(result)
    return out


# Keep the scalar qparams in the schema, matching PT2E's boundary operators.
for _name in ("quantize_per_tensor", "dequantize_per_tensor"):
    _args = "Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype"
    lib.define(f"{_name}({_args}) -> Tensor")
    lib.define(f"{_name}.out({_args}, *, Tensor(a!) out) -> Tensor(a!)")


def _check_qparams(scale, zero_point, quant_min, quant_max, dtype):
    if (
        dtype != torch.uint8
        or not math.isfinite(scale)
        or scale <= 0
        or not 0 <= quant_min <= zero_point <= quant_max <= 255
    ):
        raise ValueError("Invalid PULP uint8 quantization parameters")


@register_fake("pulp::quantize_per_tensor")
def quantize_meta(input, scale, zero_point, quant_min, quant_max, dtype):
    return torch.empty_like(input, dtype=dtype)


@register_fake("pulp::dequantize_per_tensor")
def dequantize_meta(input, scale, zero_point, quant_min, quant_max, dtype):
    return torch.empty_like(input, dtype=torch.float32)


@impl(lib, "quantize_per_tensor", "CompositeExplicitAutograd")
def quantize_impl(input, scale, zero_point, quant_min, quant_max, dtype):
    _check_qparams(scale, zero_point, quant_min, quant_max, dtype)
    if input.dtype != torch.float32:
        raise ValueError("PULP quantization expects float32 input")
    return (
        (torch.round(input * (1.0 / scale)) + zero_point)
        .clamp(quant_min, quant_max)
        .to(dtype)
    )


@impl(lib, "dequantize_per_tensor", "CompositeExplicitAutograd")
def dequantize_impl(input, scale, zero_point, quant_min, quant_max, dtype):
    _check_qparams(scale, zero_point, quant_min, quant_max, dtype)
    if input.dtype != dtype:
        raise ValueError("PULP dequantization expects uint8 input")
    return (input.to(torch.float32) - zero_point) * scale


@impl(lib, "quantize_per_tensor.out", "CompositeExplicitAutograd")
def quantize_out(input, scale, zero_point, quant_min, quant_max, dtype, *, out):
    result = quantize_impl(input, scale, zero_point, quant_min, quant_max, dtype)
    out.resize_(
        result.shape,
        memory_format=(
            torch.contiguous_format if result.dim() != 4 else torch.channels_last
        ),
    )
    out.copy_(result)
    return out


@impl(lib, "dequantize_per_tensor.out", "CompositeExplicitAutograd")
def dequantize_out(input, scale, zero_point, quant_min, quant_max, dtype, *, out):
    result = dequantize_impl(input, scale, zero_point, quant_min, quant_max, dtype)
    out.resize_(
        result.shape,
        memory_format=(
            torch.contiguous_format if result.dim() != 4 else torch.channels_last
        ),
    )
    out.copy_(result)
    return out
