# Copyright (c) 2026 KU Leuven.
"""Integer-only affine kernels and their serialized fixed-point parameters."""

from fractions import Fraction

import torch
from torch.library import impl, register_fake

from .operators import lib

INT64_MAX = (1 << 63) - 1


def fixed_point(ratios, bound=255):
    """At least eight fractional bits bound add/mean error below one output LSB."""
    for shift in range(46, 7, -1):
        multipliers = [round(Fraction(ratio) * (1 << shift)) for ratio in ratios]
        if bound * sum(abs(m) for m in multipliers) <= INT64_MAX:
            return multipliers, shift
    raise ValueError("Scale ratios exceed the supported int64 fixed-point range")


def rounded_divide(value, denominator):
    magnitude = value.abs()
    if denominator > INT64_MAX:
        rounded = (magnitude > denominator // 2).to(torch.int64)
        return torch.where(value < 0, -rounded, rounded)
    quotient = torch.div(magnitude, denominator, rounding_mode="floor")
    remainder = magnitude % denominator
    increment = (remainder > denominator // 2) | (
        (denominator % 2 == 0) & (remainder == denominator // 2) & ((quotient & 1) != 0)
    )
    rounded = quotient + increment.to(torch.int64)
    return torch.where(value < 0, -rounded, rounded)


def _params(params, size):
    if params.dtype != torch.int64 or params.shape != (size,):
        raise ValueError("Invalid integer quantization parameter tensor")
    return list(map(int, params.tolist()))


for name, arguments in (
    ("quantized_add", "Tensor self, Tensor other, Tensor params"),
    ("requantize_per_tensor", "Tensor input, Tensor params"),
    ("quantized_global_avg_pool2d", "Tensor input, Tensor params, bool keepdim=False"),
):
    lib.define(f"{name}({arguments}) -> Tensor")
    lib.define(f"{name}.out({arguments}, *, Tensor(a!) out) -> Tensor(a!)")


def _check(values, shift, zps, qmin, qmax, bound, dtype=torch.uint8):
    if dtype not in (torch.uint8, torch.int8):
        raise ValueError("Integer kernels require INT8 or UINT8")
    limits = torch.iinfo(dtype)
    if (
        not 8 <= shift <= 46
        or any(not limits.min <= z <= limits.max for z in zps)
        or not limits.min <= qmin <= zps[-1] <= qmax <= limits.max
    ):
        raise ValueError("Invalid UINT8 affine quantization parameters")
    if bound * sum(abs(m) for m in values) > INT64_MAX:
        raise ValueError("Fixed-point accumulator could overflow")


@register_fake("pulp::quantized_add")
def quantized_add_meta(self, other, params):
    return torch.empty_like(self)


@impl(lib, "quantized_add", "CompositeExplicitAutograd")
def quantized_add_impl(self, other, params):
    ma, mb, shift, za, zb, zo, qmin, qmax = _params(params, 8)
    _check((ma, mb), shift, (za, zb, zo), qmin, qmax, 255, self.dtype)
    if ma < 0:
        raise ValueError("The left input scale ratio must be nonnegative")
    if (
        self.dtype not in (torch.uint8, torch.int8)
        or other.dtype != self.dtype
        or self.shape != other.shape
        or self.dim_order() != other.dim_order()
    ):
        raise ValueError(
            "Quantized add requires same-shaped UINT8 tensors in matching layouts"
        )
    value = (self.long() - za) * ma + (other.long() - zb) * mb
    result = (rounded_divide(value, 1 << shift) + zo).clamp(qmin, qmax).to(self.dtype)
    return torch.empty_like(self).copy_(result)


@register_fake("pulp::requantize_per_tensor")
def requantize_meta(input, params):
    return torch.empty_like(input)


@impl(lib, "requantize_per_tensor", "CompositeExplicitAutograd")
def requantize_impl(input, params):
    multiplier, shift, zi, zo, qmin, qmax = _params(params, 6)
    _check((multiplier,), shift, (zi, zo), qmin, qmax, 255, input.dtype)
    if input.dtype not in (torch.uint8, torch.int8) or multiplier < 0:
        raise ValueError("Requantization requires UINT8 input and positive scale ratio")
    result = (
        (rounded_divide((input.long() - zi) * multiplier, 1 << shift) + zo)
        .clamp(qmin, qmax)
        .to(input.dtype)
    )
    return torch.empty_like(input).copy_(result)


@register_fake("pulp::quantized_global_avg_pool2d")
def quantized_pool_meta(input, params, keepdim=False):
    shape = (*input.shape[:2], 1, 1) if keepdim else input.shape[:2]
    return torch.empty(shape, dtype=input.dtype, device=input.device)


@impl(lib, "quantized_global_avg_pool2d", "CompositeExplicitAutograd")
def quantized_pool_impl(input, params, keepdim=False):
    multiplier, shift, zi, zo, qmin, qmax = _params(params, 6)
    if (
        input.dtype not in (torch.uint8, torch.int8)
        or input.dim() != 4
        or input.shape[2] <= 0
        or input.shape[3] <= 0
        or multiplier < 0
    ):
        raise ValueError("Quantized pooling requires nonempty UINT8 NCHW input")
    spatial = input.shape[2] * input.shape[3]
    _check((multiplier,), shift, (zi, zo), qmin, qmax, 255 * spatial, input.dtype)
    sums = (input.long() - zi).sum((2, 3), keepdim=keepdim) * multiplier
    denominator = spatial * (1 << shift)
    # If the denominator exceeds uint64, the bounded int64 numerator is < 1/2.
    values = (
        torch.zeros_like(sums)
        if denominator > (1 << 64) - 1
        else rounded_divide(sums, denominator)
    )
    return (values + zo).clamp(qmin, qmax).to(input.dtype).contiguous()


def _out(reference):
    def implementation(*args, out):
        result = reference(*args)
        fmt = (
            torch.channels_last
            if result.dim() == 4 and result.dim_order() == (0, 2, 3, 1)
            else torch.contiguous_format
        )
        out.resize_(result.shape, memory_format=fmt)
        out.copy_(result)
        return out

    return implementation


impl(lib, "quantized_add.out", "CompositeExplicitAutograd")(_out(quantized_add_impl))
impl(lib, "requantize_per_tensor.out", "CompositeExplicitAutograd")(
    _out(requantize_impl)
)
impl(lib, "quantized_global_avg_pool2d.out", "CompositeExplicitAutograd")(
    _out(quantized_pool_impl)
)
