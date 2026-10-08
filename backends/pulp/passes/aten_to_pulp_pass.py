# Copyright (c) 2026 ETH Zurich and University of Bologna.
# All rights reserved.

import math
from typing import cast

import executorch.backends.pulp.ops.operators  # noqa: F401
import torch
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
    get_output_qparams,
)
from executorch.backends.transforms.aten_to_dialect_pass import (
    AtenToDialectPass,
    DialectNodeSpec,
)
from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    get_param_tensor,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.graph_signature import InputKind
from torch.fx import Node


PULP_NUM_CORES = 8


class AtenToPulpPass(AtenToDialectPass):
    pass


@AtenToPulpPass.register_dialect_substitution(exir_ops.edge.aten.convolution.default)
def _get_convolution_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec | None:
    input_qparams = get_input_qparams(node)
    output_qparams = get_output_qparams(node)
    if 0 not in input_qparams or 1 not in input_qparams or not output_qparams:
        return None

    (
        input_node,
        weight_node,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        _,
        groups,
    ) = node.args
    if bias is not None:
        raise ValueError("PULP-NN MVP does not support Conv2d bias")
    if transposed or groups != 1 or tuple(dilation) != (1, 1):
        raise ValueError(
            "PULP-NN MVP supports only regular Conv2d with groups=1 and dilation=1"
        )

    input_tensor = get_first_fake_tensor(cast(Node, input_node))
    if input_tensor.dim() != 4 or input_tensor.shape[0] != 1:
        raise ValueError("PULP-NN MVP requires a static batch size of one")

    input_quant = input_qparams[0]
    weight_quant = input_qparams[1]
    output_quant = next(iter(output_qparams.values()))
    if input_quant.dtype != torch.uint8 or weight_quant.dtype != torch.int8:
        raise ValueError("PULP-NN MVP requires uint8 activations and int8 weights")
    if input_quant.zp != 0 or weight_quant.zp != 0 or output_quant.zp != 0:
        raise ValueError("PULP-NN MVP requires zero-point 0 for all tensors")

    requantize_scale = (
        float(input_quant.scale) * float(weight_quant.scale) / float(output_quant.scale)
    )
    out_shift = round(-math.log2(requantize_scale))
    if (
        out_shift < 0
        or out_shift > 31
        or not math.isclose(
            requantize_scale, 2.0**-out_shift, rel_tol=1e-5, abs_tol=1e-12
        )
    ):
        raise ValueError(
            "PULP-NN MVP requantization must be an exact power-of-two right shift; "
            f"got scale {requantize_scale}"
        )

    exported_program = dialect_pass.exported_program
    weight = get_param_tensor(exported_program, cast(Node, weight_node))
    if weight is None:
        raise ValueError("PULP-NN convolution weights must be constant")
    weight_ohwi = weight.permute(0, 2, 3, 1).contiguous()

    with node.graph.inserting_after(cast(Node, weight_node)):
        packed_weight = create_constant_placeholder(
            exported_program,
            node.graph,
            node.name + "_weight_ohwi",
            InputKind.PARAMETER,
            weight_ohwi,
        )

    _, kernel_h, kernel_w, input_channels = weight_ohwi.shape
    scratch_bytes = 2 * PULP_NUM_CORES * input_channels * kernel_h * kernel_w
    return DialectNodeSpec(
        exir_ops.edge.pulp.quantized_conv2d.default,
        (
            input_node,
            packed_weight,
            list(stride),
            list(padding),
            out_shift,
            scratch_bytes,
        ),
    )
