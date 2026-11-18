# Copyright (c) 2026 KU Leuven.
"""Solver substitutions; leave unsupported float operations portable."""

import math

import executorch.backends.pulp_dynamic.ops.operators  # noqa: F401
import torch
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
    get_output_qparams,
)
from executorch.backends.pulp.passes.aten_to_pulp_pass import _get_linear_replacement
from executorch.backends.pulp.passes.integer_region_lowering import (
    params_buffer,
    register_integer_regions,
)
from executorch.backends.pulp_dynamic.config import PulpDynamicConfig
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


class AtenToPulpDynamicPass(AtenToDialectPass):
    def __init__(self, exported_program, config=None):
        super().__init__(exported_program)
        self.config = config or PulpDynamicConfig()


def affine(node, lower, linear=False):
    if not node.meta.get("input_qparams") or not node.meta.get("output_qparams"):
        return None
    q = get_input_qparams(node)
    o = next(iter(get_output_qparams(node).values()))
    if any(v.dtype != torch.int8 or v.zp != 0 for v in (q[0], q[1], o)):
        raise ValueError("pulp_dynamic requires zero-point-zero signed INT8")
    x, wn = node.args[:2]
    bn = node.args[2] if len(node.args) > 2 else None
    if linear:
        stride, padding, groups = [1, 1], [0, 0], 1
    elif node.target == exir_ops.edge.pulp.conv2d_export.default:
        stride, padding, _, groups = node.args[3:]
    else:
        stride, padding, _, transposed, _, groups = node.args[3:]
        assert not transposed
    weight = get_param_tensor(lower.exported_program, wn)
    if weight is None or weight.dtype != torch.int8:
        raise ValueError("pulp_dynamic weights must be constant INT8")
    bias = get_param_tensor(lower.exported_program, bn) if bn is not None else None
    if bn is not None:
        if (
            bias is None
            or bias.dtype != torch.int32
            or q[2].zp != 0
            or not math.isclose(
                float(q[2].scale), float(q[0].scale) * float(q[1].scale), rel_tol=1e-5
            )
        ):
            raise ValueError("Invalid derived INT32 bias")
    limit = 128 * weight.reshape(weight.shape[0], -1).long().abs().sum(1)
    if bias is not None:
        limit = limit + bias.long().abs()
    if (limit > torch.iinfo(torch.int32).max).any():
        raise ValueError("pulp_dynamic bias plus dot product can overflow INT32")
    ratio = float(q[0].scale) * float(q[1].scale) / float(o.scale)
    shift = round(-math.log2(ratio))
    if not 0 <= shift <= 31 or not math.isclose(ratio, 2.0**-shift, rel_tol=1e-5):
        raise ValueError("Solver requires power-of-two output scaling")
    depth = not linear and groups != 1
    point = linear or (not depth and tuple(weight.shape[2:]) == (1, 1))
    kind = 2 if depth else (1 if point else 0)
    packed = (
        weight.reshape(weight.shape[0], -1).contiguous()
        if point
        else (
            weight[:, 0].contiguous()
            if depth
            else weight.permute(0, 2, 3, 1).contiguous()
        )
    )
    first = next(
        n
        for n in node.graph.nodes
        if (
            n.op == "placeholder"
            and n.name in lower.exported_program.graph_signature.user_inputs
        )
        or n.op != "placeholder"
    )
    with node.graph.inserting_before(first):
        w = create_constant_placeholder(
            lower.exported_program,
            node.graph,
            node.name + "_packed_weight",
            InputKind.BUFFER,
            packed,
            persistent_buffer=True,
        )
    settings = lower.config.serialized()
    upper = node.meta.get("pulp_activation_upper")
    if upper is not None:
        maximum = int(
            torch.round(
                torch.tensor(upper, dtype=torch.float32) * (1.0 / float(o.scale))
            )
            .clamp(0, 127)
            .item()
        )
        settings.append(maximum + 1)
    config = params_buffer(node, lower, settings)
    op = (
        exir_ops.edge.pulp_dynamic.quantized_depthwise_conv2d.default
        if depth
        else (
            exir_ops.edge.pulp_dynamic.quantized_pointwise.default
            if point
            else exir_ops.edge.pulp_dynamic.quantized_conv2d.default
        )
    )
    return DialectNodeSpec(
        op,
        (
            x,
            w,
            bn,
            list(stride),
            list(padding),
            shift,
            0,
            config,
            kind | (4 if "pulp_activation_upper" in node.meta else 0),
            linear,
        ),
    )


@AtenToPulpDynamicPass.register_dialect_substitution(
    exir_ops.edge.pulp.conv2d_export.default
)
def export_conv(node, lower):
    result = affine(node, lower)
    if result is not None:
        return result
    x, w, b, s, p, d, g = node.args
    return DialectNodeSpec(
        exir_ops.edge.aten.convolution.default, (x, w, b, s, p, d, False, [0, 0], g)
    )


@AtenToPulpDynamicPass.register_dialect_substitution(
    exir_ops.edge.aten.convolution.default
)
def conv(node, lower):
    return affine(node, lower)


@AtenToPulpDynamicPass.register_dialect_substitution(exir_ops.edge.aten.linear.default)
def linear(node, lower):
    result = affine(node, lower, True)
    if result is not None:
        return result
    return _get_linear_replacement(node, lower)


for source, destination in (
    (
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.pulp.quantize_per_tensor_cluster.default,
    ),
    (
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        exir_ops.edge.pulp.dequantize_per_tensor_cluster.default,
    ),
):

    def conversion(node, lower, destination=destination):
        if node.args[5] not in (torch.int8, torch.uint8):
            return None
        return DialectNodeSpec(destination, node.args)

    AtenToPulpDynamicPass.register_dialect_substitution(source)(conversion)
register_integer_regions(AtenToPulpDynamicPass)
