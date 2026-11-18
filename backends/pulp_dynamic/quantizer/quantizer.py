# Copyright (c) 2026 KU Leuven.
"""Conservative eligibility before Q/DQ folding; report every float fallback."""

import math

import torch
from executorch.backends.arm.quantizer.arm_quantizer_utils import PatternCheck
from executorch.backends.pulp.quantizer.quantizer import (
    PulpConv2DCheck,
    PulpLinearCheck,
    PulpQuantizer,
)
from executorch.backends.pulp_dynamic.config import PulpDynamicConfig

from .quantization_config import CONV_CONFIG, REGION_CONFIG


def support_reason(node, config):
    linear = node.target == torch.ops.aten.linear.default
    values = [node.args[i].meta.get("val") for i in (0, 1)]
    x, w = values
    if linear:
        if not PulpLinearCheck.check_pattern([node]):
            return "linear requires contiguous input and constant-shaped 2-D weights"
        ci, kh, kw, pad, depth = x.shape[-1], 1, 1, (0, 0), False
    else:
        if not PulpConv2DCheck.check_pattern([node]):
            return "requires batch=1, channels-last, standard/multiplier-one depthwise and dilation=1"
        ci = x.shape[1]
        kh, kw = w.shape[2:]
        stride = node.args[3] if len(node.args) > 3 else (1, 1)
        pad = node.args[4] if len(node.args) > 4 else (0, 0)
        groups = node.args[6] if len(node.args) > 6 else 1
        depth = groups != 1
        if (kh, kw) == (1, 1) and (tuple(stride) != (1, 1) or tuple(pad) != (0, 0)):
            return "pointwise primitive requires stride=1 and padding=0"
    if not linear:
        for size in x.shape[2:]:
            upper = (
                size.node.shape_env.bound_sympy(size.node.expr).upper
                if isinstance(size, torch.SymInt)
                else size
            )
            try:
                if int(upper) > 65535:
                    return "maximum spatial dimension exceeds primitive uint16 limit"
            except (ValueError, OverflowError):
                return "spatial dimensions require finite exported upper bounds"
    if any(
        not isinstance(v, int) or not 0 < v <= 65535 for v in (ci, kh, kw, w.shape[0])
    ):
        return "channel/kernel dimensions must be static and fit uint16"
    reduction = kh * kw * (1 if depth else ci)
    if 128 * 127 * reduction > torch.iinfo(torch.int32).max:
        return "dot product exceeds INT32"
    tile_ci = 1 if depth else ci
    # Conservative minimum tile, with all Ci retained. Include alignment and
    # the solver's scratch reservation even for scalar depthwise execution.
    scratch = (
        (config.num_cores * kw * (kh + 2 * pad[0]) + config.num_cores * kw)
        if depth
        else (0 if (kh, kw) == (1, 1) else 2 * config.num_cores * kh * kw * ci)
    )
    minimum = (
        (kh + 2 * pad[0]) * (kw + 2 * pad[1]) * tile_ci
        + kh * kw * tile_ci
        + 1
        + 12
        + scratch
        + 16
    )
    if minimum * (2 if config.double_buffering else 1) > config.l1_bytes:
        return "minimum tile with full Ci exceeds configured L1 (Ci tiling unsupported)"
    return None


class PulpDynamicQuantizer(PulpQuantizer):
    def __init__(self, config=None):
        self.config = config or PulpDynamicConfig()
        config = self.config

        class DynamicCheck(PatternCheck):
            @classmethod
            def check_pattern(cls, pattern):
                return len(pattern) == 1 and support_reason(pattern[0], config) is None

            @classmethod
            def check_quantization_config(cls, pattern, quantization_config):
                return True

        self.patterns = {
            (op,): DynamicCheck
            for op in (
                torch.ops.aten.conv2d.default,
                torch.ops.pulp.conv2d_export.default,
                torch.ops.aten.linear.default,
            )
        }

        class ConvActivationCheck(DynamicCheck):
            @classmethod
            def check_pattern(cls, pattern):
                if len(pattern) != 2:
                    return False
                conv, activation = pattern
                if activation.target == torch.ops.aten.hardtanh.default:
                    lo = activation.args[1] if len(activation.args) > 1 else -1.0
                    hi = activation.args[2] if len(activation.args) > 2 else 1.0
                    if lo != 0 or hi <= 0 or not math.isfinite(hi):
                        return False
                return (
                    activation.args[0] is conv
                    and len(conv.users) == 1
                    and support_reason(conv, config) is None
                )

        for op in (
            torch.ops.aten.conv2d.default,
            torch.ops.pulp.conv2d_export.default,
            torch.ops.aten.linear.default,
        ):
            for activation in (
                torch.ops.aten.relu.default,
                torch.ops.aten.relu6.default,
                torch.ops.aten.hardtanh.default,
            ):
                self.patterns[(op, activation)] = ConvActivationCheck
        self.conv_config = CONV_CONFIG
        self.region_config = REGION_CONFIG
        super().__init__()

    def transform_for_annotation(self, model):
        for n in model.graph.nodes:
            if n.target in (
                torch.ops.aten.conv2d.default,
                torch.ops.pulp.conv2d_export.default,
                torch.ops.aten.linear.default,
            ):
                reason = support_reason(n, self.config)
                if reason:
                    print(
                        f"PORTABLE FALLBACK: node={n.name} op={n.target} reason={reason}",
                        flush=True,
                    )
        return super().transform_for_annotation(model)
