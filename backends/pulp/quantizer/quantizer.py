# Copyright (c) 2026 KU Leuven.
# All rights reserved.

import math
from typing import cast

import executorch.backends.pulp.ops.operators  # noqa: F401

import torch
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm.quantizer.arm_quantizer_utils import (
    PatternCheck,
    PatternQuantizer,
)
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from executorch.backends.cortex_m.quantizer.node_finders import NodeTargetNodeFinder
from executorch.backends.cortex_m.quantizer.pattern_matcher import PatternMatcher
from executorch.backends.pulp.quantizer.quantization_config import PULP_CONV_CONFIG
from torch._ops import OpOverload
from torch.fx import GraphModule, Node
from torchao.quantization.pt2e.quantizer import ComposableQuantizer, Quantizer


class PulpConv2DCheck(PatternCheck):
    @classmethod
    def check_pattern(cls, pattern: list[Node]) -> bool:
        if len(pattern) != 1:
            return False
        node = pattern[0]
        tensor = get_first_fake_tensor(node)
        if tensor.dim() != 4:
            return False
        if tensor.shape[0] != 1 or not tensor.is_contiguous(
            memory_format=torch.channels_last
        ):
            return False
        input_tensor = get_first_fake_tensor(cast(Node, node.args[0]))
        weight_tensor = get_first_fake_tensor(cast(Node, node.args[1]))
        if input_tensor.dim() != 4 or weight_tensor.dim() != 4:
            return False
        if not input_tensor.is_contiguous(memory_format=torch.channels_last):
            return False
        in_channels = input_tensor.shape[1]
        out_channels = weight_tensor.shape[0]
        dilation = node.args[5] if len(node.args) > 5 else (1, 1)
        groups = node.args[6] if len(node.args) > 6 else 1
        is_standard = groups == 1 and weight_tensor.shape[1] == in_channels
        is_depthwise = (
            groups == in_channels
            and out_channels == in_channels
            and weight_tensor.shape[1] == 1
        )
        return tuple(dilation) == (1, 1) and (is_standard or is_depthwise)

    @classmethod
    def check_quantization_config(
        cls, pattern: list[Node], quantization_config: QuantizationConfig
    ) -> bool:
        del pattern
        input_qspec = quantization_config.get_input_act_qspec()
        weight_qspec = quantization_config.get_weight_qspec()
        return (
            input_qspec is not None
            and weight_qspec is not None
            and input_qspec.dtype == torch.uint8
            and weight_qspec.dtype == torch.int8
        )


class PulpLinearCheck(PulpConv2DCheck):
    @classmethod
    def check_pattern(cls, pattern: list[Node]) -> bool:
        if len(pattern) != 1:
            return False
        node = pattern[0]
        inputs = get_first_fake_tensor(cast(Node, node.args[0]))
        weights = get_first_fake_tensor(cast(Node, node.args[1]))
        return (
            inputs.dim() >= 1
            and inputs.is_contiguous()
            and weights.dim() == 2
            and inputs.shape[-1] == weights.shape[1]
            and weights.shape[0] > 0
            and weights.shape[1] > 0
        )


class PulpConvActivationCheck(PulpConv2DCheck):
    @classmethod
    def check_pattern(cls, pattern: list[Node]) -> bool:
        if len(pattern) != 2 or len(pattern[0].users) != 1:
            return False
        conv, activation = pattern
        check = (
            PulpLinearCheck
            if conv.target == torch.ops.aten.linear.default
            else PulpConv2DCheck
        )
        if not check.check_pattern([conv]) or activation.args[0] is not conv:
            return False
        if activation.target == torch.ops.aten.hardtanh.default:
            lower = activation.args[1] if len(activation.args) > 1 else -1.0
            upper = activation.args[2] if len(activation.args) > 2 else 1.0
            # Only nonnegative clipping is compatible with the unsigned kernels.
            return lower == 0 and upper > 0 and math.isfinite(upper)
        return True


PULP_PATTERNS = {
    (torch.ops.aten.conv2d.default,): PulpConv2DCheck,
    (torch.ops.pulp.conv2d_export.default,): PulpConv2DCheck,
    (torch.ops.aten.linear.default,): PulpLinearCheck,
}

for _affine in (
    torch.ops.aten.conv2d.default,
    torch.ops.pulp.conv2d_export.default,
    torch.ops.aten.linear.default,
):
    for _activation in (
        torch.ops.aten.relu.default,
        torch.ops.aten.relu6.default,
        torch.ops.aten.hardtanh.default,
    ):
        PULP_PATTERNS[(_affine, _activation)] = PulpConvActivationCheck


class PulpPatternQuantizer(PatternQuantizer):
    PARAMETER_TARGETS = PatternQuantizer.PARAMETER_TARGETS | {
        torch.ops.pulp.conv2d_export.default
    }


class PulpQuantizer(ComposableQuantizer):
    def __init__(self) -> None:
        targets: set[OpOverload] = set()
        for pattern in PULP_PATTERNS:
            targets.update(pattern)
        matcher = PatternMatcher(
            PULP_PATTERNS, support_dict_name=__name__ + ".PULP_PATTERNS"
        )
        quantizers: list[Quantizer] = [
            PulpPatternQuantizer(
                PULP_CONV_CONFIG,
                node_finder=NodeTargetNodeFinder(list(targets)),
                pattern_matcher=matcher,
            )
        ]
        super().__init__(quantizers)

    def validate(self, model: GraphModule) -> None:
        return None

    def transform_for_annotation(self, model: GraphModule) -> GraphModule:
        return cast(GraphModule, model)
