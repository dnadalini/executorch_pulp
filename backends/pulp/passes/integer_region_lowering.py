# Copyright (c) 2026 KU Leuven.
"""Replace annotated UINT8 regions and prepare constant fixed-point parameters."""

from fractions import Fraction

import torch
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
    get_output_qparams,
)
from executorch.backends.pulp.ops.integer_ops import fixed_point
from executorch.backends.transforms.aten_to_dialect_pass import DialectNodeSpec
from executorch.backends.transforms.utils import create_constant_placeholder
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.graph_signature import InputKind


def params_buffer(node, dialect_pass, values):
    first_user = next(
        n
        for n in node.graph.nodes
        if (
            n.op == "placeholder"
            and n.name in dialect_pass.exported_program.graph_signature.user_inputs
        )
        or n.op != "placeholder"
    )
    with node.graph.inserting_before(first_user):
        return create_constant_placeholder(
            dialect_pass.exported_program,
            node.graph,
            node.name + "_integer_params",
            InputKind.BUFFER,
            torch.tensor(values, dtype=torch.int64),
            persistent_buffer=True,
        )


def _fraction(value):
    return Fraction(float(value))


def _maximum_spatial(value):
    total = 1
    for size in value.shape[2:]:
        upper = (
            size.node.shape_env.bound_sympy(size.node.expr).upper
            if isinstance(size, torch.SymInt)
            else size
        )
        total *= int(upper)
    return total


def _float_fallback(node, inputs, output, alpha=None):
    """Preserve the full affine contract if coefficients cannot safely fit int64."""
    with node.graph.inserting_before(node):
        tensors = [
            node.graph.call_function(
                exir_ops.edge.pulp.dequantize_per_tensor_cluster.default,
                (
                    node.args[i],
                    float(q.scale),
                    int(q.zp),
                    int(q.qmin),
                    int(q.qmax),
                    q.dtype,
                ),
            )
            for i, q in enumerate(inputs)
        ]
        if alpha is None:
            value = node.graph.call_function(
                exir_ops.edge.pulp.global_avg_pool2d.default,
                (tensors[0], node.args[1] if len(node.args) > 1 else False),
            )
        else:
            value = node.graph.call_function(
                exir_ops.edge.pulp.add_cluster.default, (*tensors, alpha)
            )
    return DialectNodeSpec(
        exir_ops.edge.pulp.quantize_per_tensor_cluster.default,
        (
            value,
            float(output.scale),
            int(output.zp),
            int(output.qmin),
            int(output.qmax),
            output.dtype,
        ),
    )


def register_integer_regions(pass_class):
    @pass_class.register_dialect_substitution(exir_ops.edge.pulp.add_cluster.default)
    def add(node, dialect_pass):
        if not node.meta.get("input_qparams") or not node.meta.get("output_qparams"):
            return None
        inputs, outputs = get_input_qparams(node), get_output_qparams(node)
        if 0 not in inputs or 1 not in inputs:
            raise ValueError(
                "Quantized residual add needs both input quantization parameters"
            )
        qa, qb, qo = inputs[0], inputs[1], next(iter(outputs.values()))
        if qa.dtype not in (torch.uint8, torch.int8) or any(
            q.dtype != qa.dtype for q in (qb, qo)
        ):
            raise ValueError("PULP integer regions require UINT8 quantization")
        alpha = float(
            torch.tensor(
                node.args[2] if len(node.args) > 2 else 1.0, dtype=torch.float32
            )
        )
        try:
            multipliers, shift = fixed_point(
                (
                    _fraction(qa.scale) / _fraction(qo.scale),
                    _fraction(alpha) * _fraction(qb.scale) / _fraction(qo.scale),
                )
            )
        except (ValueError, OverflowError, ZeroDivisionError):
            return _float_fallback(node, (qa, qb), qo, alpha)
        parameters = params_buffer(
            node,
            dialect_pass,
            [
                *multipliers,
                shift,
                int(qa.zp),
                int(qb.zp),
                int(qo.zp),
                int(qo.qmin),
                int(qo.qmax),
            ],
        )
        return DialectNodeSpec(
            exir_ops.edge.pulp.quantized_add.default, (*node.args[:2], parameters)
        )

    @pass_class.register_dialect_substitution(
        exir_ops.edge.pulp.global_avg_pool2d.default
    )
    def pool(node, dialect_pass):
        if not node.meta.get("input_qparams") or not node.meta.get("output_qparams"):
            return None
        qi = get_input_qparams(node)[0]
        qo = next(iter(get_output_qparams(node).values()))
        if qi.dtype not in (torch.uint8, torch.int8) or qo.dtype != qi.dtype:
            raise ValueError("PULP integer pooling requires UINT8 quantization")
        try:
            spatial = _maximum_spatial(node.args[0].meta["val"])
            multipliers, shift = fixed_point(
                (_fraction(qi.scale) / _fraction(qo.scale),), 255 * spatial
            )
        except (ValueError, OverflowError):
            return _float_fallback(node, (qi,), qo)
        parameters = params_buffer(
            node,
            dialect_pass,
            [*multipliers, shift, int(qi.zp), int(qo.zp), int(qo.qmin), int(qo.qmax)],
        )
        return DialectNodeSpec(
            exir_ops.edge.pulp.quantized_global_avg_pool2d.default,
            (node.args[0], parameters, node.args[1] if len(node.args) > 1 else False),
        )
