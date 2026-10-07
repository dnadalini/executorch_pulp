# Copyright (c) 2026 ETH Zurich and University of Bologna.
# All rights reserved.

import math
from typing import Any

import torch
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from torch.fx import Node
from torchao.quantization.pt2e import MinMaxObserver
from torchao.quantization.pt2e.quantizer import (
    DerivedQuantizationSpec,
    QuantizationSpec,
    QuantizationSpecBase,
)


class PulpUnsignedMinMaxObserver(MinMaxObserver):
    """Unsigned observer whose real zero is represented by byte zero."""

    @torch.jit.export
    def calculate_qparams(self):
        max_val = torch.maximum(self.max_val, torch.zeros_like(self.max_val))
        scale = torch.maximum(
            max_val / 255.0,
            torch.as_tensor(self.eps, device=max_val.device, dtype=max_val.dtype),
        ).reshape(1)
        zero_point = torch.zeros(1, dtype=torch.int64, device=max_val.device)
        return scale, zero_point


PULP_UINT8_ACTIVATION_QSPEC = QuantizationSpec(
    dtype=torch.uint8,
    observer_or_fake_quant_ctr=PulpUnsignedMinMaxObserver,
    quant_min=0,
    quant_max=255,
    qscheme=torch.per_tensor_affine,
)

PULP_INT8_WEIGHT_QSPEC = QuantizationSpec(
    dtype=torch.int8,
    observer_or_fake_quant_ctr=MinMaxObserver,
    quant_min=-127,
    quant_max=127,
    qscheme=torch.per_tensor_symmetric,
)


class PulpConvQuantizationConfig(QuantizationConfig):
    def get_output_act_qspec(
        self, node: Node | None = None
    ) -> QuantizationSpecBase | None:
        if node is None or len(node.args) < 2:
            return None

        weight = node.args[1]
        if not isinstance(weight, Node):
            return None
        weight_value = weight.meta.get("val")
        if not isinstance(weight_value, torch.Tensor) or weight_value.dim() != 4:
            return None
        accumulator_terms = math.prod(weight_value.shape[1:])
        if 255 * 127 * accumulator_terms > torch.iinfo(torch.int32).max:
            raise ValueError("PULP-NN Conv2d exceeds the int32 accumulator range")
        requant_shift = math.ceil(math.log2(127 * accumulator_terms))

        def derive_output_qparams(obs_or_fqs: list[Any]):
            if len(obs_or_fqs) != 2:
                raise ValueError("PULP Conv2d output requires input and weight observers")
            input_scale, _ = obs_or_fqs[0].calculate_qparams()
            weight_scale, _ = obs_or_fqs[1].calculate_qparams()
            output_scale = input_scale * weight_scale * (1 << requant_shift)
            return output_scale.to(torch.float32), torch.zeros_like(
                output_scale, dtype=torch.int64
            )

        return DerivedQuantizationSpec(
            derived_from=[(node.args[0], node), (node.args[1], node)],
            derive_qparams_fn=derive_output_qparams,
            dtype=torch.uint8,
            quant_min=0,
            quant_max=255,
            qscheme=torch.per_tensor_affine,
        )


PULP_CONV_CONFIG = PulpConvQuantizationConfig(
    input_activation=PULP_UINT8_ACTIVATION_QSPEC,
    output_activation=PULP_UINT8_ACTIVATION_QSPEC,
    weight=PULP_INT8_WEIGHT_QSPEC,
    bias=None,
    label="PULP-NN u8s8 Conv2d",
)
