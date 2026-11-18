# Copyright (c) 2026 KU Leuven.
"""Signed reference for the solver's shift-and-saturate kernels."""

import math

import torch
import torch.nn.functional as F
from torch.library import impl, Library, register_fake

lib = Library("pulp_dynamic", "DEF")
ARGS = "Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int out_shift, int layer_index, Tensor config, int kind=0, bool linear=False"


def meta(
    input,
    weight,
    bias,
    stride,
    padding,
    out_shift,
    layer_index,
    config,
    kind=0,
    linear=False,
):
    kind = kind & 3
    if linear:
        return torch.empty(
            (*input.shape[:-1], weight.shape[0]), dtype=torch.int8, device=input.device
        )
    kh, kw = (
        weight.shape[1:3] if kind == 0 else ((1, 1) if kind == 1 else weight.shape[1:])
    )
    h = (input.shape[2] + 2 * padding[0] - kh) // stride[0] + 1
    w = (input.shape[3] + 2 * padding[1] - kw) // stride[1] + 1
    return torch.empty(
        (input.shape[0], weight.shape[0], h, w),
        dtype=torch.int8,
        device=input.device,
        memory_format=torch.channels_last,
    )


def reference(
    input,
    weight,
    bias,
    stride,
    padding,
    out_shift,
    layer_index,
    config,
    kind=0,
    linear=False,
):
    fused_relu = bool(kind & 4)
    kind = kind & 3
    if (
        input.dtype != torch.int8
        or weight.dtype != torch.int8
        or not 0 <= out_shift <= 31
    ):
        raise ValueError(
            "pulp_dynamic requires signed INT8 and a right shift in [0,31]"
        )
    if config.dtype != torch.int64 or config.numel() not in (6, 7):
        raise ValueError("Invalid deployment config")
    maximum = int(config[6]) - 1 if config.numel() == 7 else 127
    if not 0 <= maximum <= 127 or (config.numel() == 7 and not fused_relu):
        raise ValueError("Invalid fused activation bound")
    if bias is not None and (
        bias.dtype != torch.int32 or bias.shape != (weight.shape[0],)
    ):
        raise ValueError("Invalid INT32 bias")
    if linear:
        value = F.linear(
            input.double(), weight.double(), None if bias is None else bias.double()
        ).long()
        return (
            (value >> out_shift)
            .clamp(0 if fused_relu else -128, maximum)
            .to(torch.int8)
        )
    w = (
        weight.permute(0, 3, 1, 2)
        if kind == 0
        else (
            weight.reshape(weight.shape[0], weight.shape[1], 1, 1)
            if kind == 1
            else weight.unsqueeze(1)
        )
    )
    value = F.conv2d(
        input.double(),
        w.double(),
        None if bias is None else bias.double(),
        stride,
        padding,
        groups=input.shape[1] if kind == 2 else 1,
    ).long()
    return (
        (value >> out_shift)
        .clamp(0 if fused_relu else -128, maximum)
        .to(torch.int8, memory_format=torch.channels_last)
    )


def out_impl(*args, out):
    value = reference(*args)
    out.resize_(
        value.shape,
        memory_format=(
            torch.channels_last
            if value.dim() == 4 and value.dim_order() == (0, 2, 3, 1)
            else torch.contiguous_format
        ),
    )
    out.copy_(value)
    return out


for name in ("quantized_conv2d", "quantized_pointwise", "quantized_depthwise_conv2d"):
    lib.define(f"{name}({ARGS}) -> Tensor")
    lib.define(f"{name}.out({ARGS}, *, Tensor(a!) out) -> Tensor(a!)")
    register_fake("pulp_dynamic::" + name)(meta)
    impl(lib, name, "CompositeExplicitAutograd")(reference)
    impl(lib, name + ".out", "CompositeExplicitAutograd")(out_impl)
