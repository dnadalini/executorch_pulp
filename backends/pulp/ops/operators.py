# Copyright (c) 2026 ETH Zurich and University of Bologna.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch.library import impl, Library, register_fake


lib = Library("pulp", "DEF")

lib.define(
    "quantized_conv2d(Tensor input, Tensor weight, int[] stride, int[] padding, "
    "int out_shift, int scratch_bytes) -> Tensor"
)
lib.define(
    "quantized_conv2d.out(Tensor input, Tensor weight, int[] stride, int[] padding, "
    "int out_shift, int scratch_bytes, *, Tensor(a!) out) -> Tensor(a!)"
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
    stride: Sequence[int],
    padding: Sequence[int],
    out_shift: int,
    scratch_bytes: int,
) -> torch.Tensor:
    del out_shift, scratch_bytes
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
    stride: Sequence[int],
    padding: Sequence[int],
    out_shift: int,
    scratch_bytes: int,
) -> torch.Tensor:
    del scratch_bytes
    if input.dtype != torch.uint8 or weight.dtype != torch.int8:
        raise ValueError("PULP-NN convolution expects uint8 input and int8 weight")
    if input.dim() != 4 or weight.dim() != 4 or input.shape[0] != 1:
        raise ValueError("PULP-NN convolution expects 4-D tensors and batch size 1")

    weight_oihw = weight.permute(0, 3, 1, 2).contiguous()
    accumulator = F.conv2d(
        input.to(torch.float32),
        weight_oihw.to(torch.float32),
        bias=None,
        stride=tuple(stride),
        padding=tuple(padding),
    ).to(torch.int32)
    output = torch.bitwise_right_shift(accumulator, out_shift).clamp(0, 255)
    return output.to(torch.uint8, memory_format=torch.channels_last)


@impl(lib, "quantized_conv2d.out", "CompositeExplicitAutograd")
def quantized_conv2d_out_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride: Sequence[int],
    padding: Sequence[int],
    out_shift: int,
    scratch_bytes: int,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    out.copy_(
        quantized_conv2d_impl(
            input, weight, stride, padding, out_shift, scratch_bytes
        )
    )
    return out
