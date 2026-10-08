# Copyright (c) 2026 ETH Zurich and University of Bologna.
# All rights reserved.

from typing import cast

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
        bias = node.args[2] if len(node.args) > 2 else None
        dilation = node.args[5] if len(node.args) > 5 else (1, 1)
        groups = node.args[6] if len(node.args) > 6 else 1
        return bias is None and tuple(dilation) == (1, 1) and groups == 1

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


PULP_PATTERNS = {(torch.ops.aten.conv2d.default,): PulpConv2DCheck}


class PulpQuantizer(ComposableQuantizer):
    def __init__(self) -> None:
        targets: set[OpOverload] = set()
        for pattern in PULP_PATTERNS:
            targets.update(pattern)
        matcher = PatternMatcher(
            PULP_PATTERNS, support_dict_name=__name__ + ".PULP_PATTERNS"
        )
        quantizers: list[Quantizer] = [
            PatternQuantizer(
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
