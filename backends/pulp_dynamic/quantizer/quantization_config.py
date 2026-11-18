# Copyright (c) 2026 KU Leuven.
import math

import torch
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from executorch.backends.pulp.quantizer.quantization_config import (
    PULP_INT8_WEIGHT_QSPEC,
    PulpConvQuantizationConfig,
)
from torchao.quantization.pt2e import MinMaxObserver
from torchao.quantization.pt2e.quantizer import (
    DerivedQuantizationSpec,
    QuantizationSpec,
)

SIGNED_ACTIVATION = QuantizationSpec(
    dtype=torch.int8,
    observer_or_fake_quant_ctr=MinMaxObserver,
    quant_min=-128,
    quant_max=127,
    qscheme=torch.per_tensor_symmetric,
)


class DynamicConvConfig(PulpConvQuantizationConfig):
    def get_output_act_qspec(self, node=None):
        if node is None:
            return None
        if node.target in (
            torch.ops.aten.relu.default,
            torch.ops.aten.relu6.default,
            torch.ops.aten.hardtanh.default,
        ):
            node = node.args[0]
        weight = node.args[1].meta["val"]
        terms = math.prod(weight.shape[1:])
        if 128 * 127 * terms > torch.iinfo(torch.int32).max:
            raise ValueError("pulp_dynamic accumulator exceeds INT32")
        shift = math.ceil(math.log2(128 * terms))

        def derive(observers):
            si, _ = observers[0].calculate_qparams()
            sw, _ = observers[1].calculate_qparams()
            so = (si * sw * (1 << shift)).float()
            return so, torch.zeros_like(so, dtype=torch.int64)

        return DerivedQuantizationSpec(
            derived_from=[(node.args[0], node), (node.args[1], node)],
            derive_qparams_fn=derive,
            dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            qscheme=torch.per_tensor_symmetric,
        )


CONV_CONFIG = DynamicConvConfig(
    input_activation=SIGNED_ACTIVATION,
    output_activation=SIGNED_ACTIVATION,
    weight=PULP_INT8_WEIGHT_QSPEC,
    bias=None,
    label="pulp_dynamic s8s8",
)
REGION_CONFIG = QuantizationConfig(
    input_activation=SIGNED_ACTIVATION,
    output_activation=SIGNED_ACTIVATION,
    weight=None,
    bias=None,
    label="pulp_dynamic signed residual/pool",
)
