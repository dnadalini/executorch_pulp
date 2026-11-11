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
    return _get_affine_replacement(node, dialect_pass)


@AtenToPulpPass.register_dialect_substitution(exir_ops.edge.aten.linear.default)
def _get_linear_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec | None:
    replacement = _get_affine_replacement(node, dialect_pass, linear=True)
    if replacement is not None:
        return replacement
    # Preserve float fallback when a linear was deliberately left unquantized.
    # Edge preserves linear for PULP lowering, so decompose only this remainder.
    inputs, weights = node.args[:2]
    bias = node.args[2] if len(node.args) > 2 else None
    input_tensor = get_first_fake_tensor(cast(Node, inputs))
    weight_tensor = get_first_fake_tensor(cast(Node, weights))
    with node.graph.inserting_before(node):

        def size_value(value, tensor, dim):
            size = tensor.shape[dim]
            return (
                size
                if isinstance(size, int)
                else node.graph.call_function(
                    exir_ops.edge.aten.sym_size.int, (value, dim)
                )
            )

        contiguous = node.graph.call_function(
            exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
            (inputs,),
            {"dim_order": list(range(input_tensor.dim()))},
        )
        flat = node.graph.call_function(
            exir_ops.edge.aten.view_copy.default,
            (contiguous, [-1, size_value(weights, weight_tensor, 1)]),
        )
        transposed = node.graph.call_function(
            exir_ops.edge.aten.permute_copy.default, (weights, [1, 0])
        )
        if bias is None:
            result = node.graph.call_function(
                exir_ops.edge.aten.mm.default, (flat, transposed)
            )
        else:
            result = node.graph.call_function(
                exir_ops.edge.aten.addmm.default, (bias, flat, transposed)
            )
        shape = [
            size_value(inputs, input_tensor, dim)
            for dim in range(input_tensor.dim() - 1)
        ]
        shape.append(size_value(weights, weight_tensor, 0))
    return DialectNodeSpec(exir_ops.edge.aten.view_copy.default, (result, shape))


def _get_affine_replacement(
    node: Node, dialect_pass: AtenToDialectPass, linear: bool = False
) -> DialectNodeSpec | None:
    # Nodes without folded quantization parameters stay on the fallback path.
    if not node.meta.get("input_qparams") or not node.meta.get("output_qparams"):
        return None

    input_qparams = get_input_qparams(node)
    output_qparams = get_output_qparams(node)
    if 0 not in input_qparams or 1 not in input_qparams or not output_qparams:
        return None
    if linear:
        input_node, weight_node = node.args[:2]
        bias = node.args[2] if len(node.args) > 2 else None
        stride, padding, dilation, transposed, groups = (1, 1), (0, 0), (1, 1), False, 1
    else:
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
    input_tensor = get_first_fake_tensor(cast(Node, input_node))
    weight_tensor = get_first_fake_tensor(cast(Node, weight_node))
    if linear:
        if (
            input_tensor.dim() < 1
            or not input_tensor.is_contiguous()
            or weight_tensor.dim() != 2
        ):
            raise ValueError("PULP-NN linear requires contiguous input and 2-D weights")
        in_channels = input_tensor.shape[-1]
    else:
        if (
            input_tensor.dim() != 4
            or input_tensor.shape[0] != 1
            or weight_tensor.dim() != 4
        ):
            raise ValueError(
                "PULP-NN convolution requires a 4-D input with batch size one and 4-D weights"
            )
        in_channels = input_tensor.shape[1]
    out_channels = weight_tensor.shape[0]
    is_standard = groups == 1
    is_depthwise = not linear and groups != 1 and groups == in_channels == out_channels
    if transposed or not (is_standard or is_depthwise) or tuple(dilation) != (1, 1):
        raise ValueError(
            "PULP-NN supports standard/depthwise convolution with dilation=1"
        )
    if weight_tensor.shape[1] != (1 if is_depthwise else in_channels):
        raise ValueError("PULP-NN received incompatible weights")

    input_quant = input_qparams[0]
    weight_quant = input_qparams[1]
    output_quant = next(iter(output_qparams.values()))

    if (
        input_quant.dtype != torch.uint8
        or weight_quant.dtype != torch.int8
        or output_quant.dtype != torch.uint8
    ):
        raise ValueError("PULP-NN requires uint8 activations/output and int8 weights")
    if input_quant.zp != 0 or weight_quant.zp != 0 or output_quant.zp != 0:
        raise ValueError("PULP-NN requires zero-point 0 for all tensors")

    input_scale = float(input_quant.scale)
    weight_scale = float(weight_quant.scale)
    output_scale = float(output_quant.scale)
    if any(
        not math.isfinite(scale) or scale <= 0
        for scale in (input_scale, weight_scale, output_scale)
    ):
        raise ValueError("PULP-NN requires finite positive quantization scales")

    requantize_scale = input_scale * weight_scale / output_scale
    if not math.isfinite(requantize_scale) or requantize_scale <= 0:
        raise ValueError("PULP-NN received an invalid requantization scale")

    out_shift = round(-math.log2(requantize_scale))
    if not 0 <= out_shift <= 31 or not math.isclose(
        requantize_scale, 2.0**-out_shift, rel_tol=1e-5
    ):
        raise ValueError(
            "PULP-NN requantization must be an exact power-of-two "
            f"right shift; got scale {requantize_scale}"
        )

    exported_program = dialect_pass.exported_program
    weight = get_param_tensor(exported_program, cast(Node, weight_node))
    if weight is None:
        raise ValueError("PULP-NN convolution weights must be constant")
    if weight.dtype != torch.int8 or tuple(weight.shape) != tuple(weight_tensor.shape):
        raise ValueError("PULP-NN received incompatible constant weights")

    kernel_h, kernel_w = (1, 1) if linear else weight.shape[2:]
    is_pointwise = linear or (is_standard and (kernel_h, kernel_w) == (1, 1))

    if is_pointwise:
        packed_weight_tensor = weight.reshape(out_channels, in_channels).contiguous()
        weight_layout = "oi"
        replacement_op = exir_ops.edge.pulp.quantized_pointwise.default
    elif is_depthwise:
        packed_weight_tensor = weight.contiguous()
        weight_layout = "oihw"
        replacement_op = exir_ops.edge.pulp.quantized_depthwise_conv2d.default
    else:
        packed_weight_tensor = weight.permute(0, 2, 3, 1).contiguous()
        weight_layout = "ohwi"
        replacement_op = exir_ops.edge.pulp.quantized_conv2d.default

    bias_tensor = None
    if bias is not None:
        bias_tensor = get_param_tensor(exported_program, cast(Node, bias))
        bias_quant = input_qparams.get(2)
        if (
            bias_tensor is None
            or bias_tensor.dtype != torch.int32
            or tuple(bias_tensor.shape) != (weight.shape[0],)
            or bias_quant is None
            or bias_quant.zp != 0
            or not math.isclose(
                float(bias_quant.scale),
                input_scale * weight_scale,
                rel_tol=1e-5,
            )
        ):
            raise ValueError(
                "PULP-NN bias must be constant int32 with scale input_scale * weight_scale"
            )
        if (bias_tensor <= -(2**31) + 256).any() or (bias_tensor >= 2**31 - 256).any():
            raise ValueError(
                "PULP-NN bias exceeds the supported int32 quantization range"
            )

    # These bounds apply independently to each output channel,
    # including depthwise convolution.
    weight_i64 = weight.reshape(weight.shape[0], -1).to(torch.int64)
    initial = bias_tensor.to(torch.int64) if bias_tensor is not None else 0
    lower = initial + 255 * weight_i64.clamp(max=0).sum(1)
    upper = initial + 255 * weight_i64.clamp(min=0).sum(1)
    if (lower < -(2**31)).any() or (upper > 2**31 - 1).any():
        raise ValueError("PULP-NN bias plus convolution may overflow int32")

    with node.graph.inserting_after(cast(Node, weight_node)):
        packed_weight = create_constant_placeholder(
            exported_program,
            node.graph,
            node.name + f"_weight_{weight_layout}",
            InputKind.BUFFER,
            packed_weight_tensor,
            persistent_buffer=True,
        )
        scales = create_constant_placeholder(
            exported_program,
            node.graph,
            node.name + "_scales",
            InputKind.BUFFER,
            torch.tensor(
                [input_scale, weight_scale, output_scale],
                dtype=torch.float32,
            ),
            persistent_buffer=True,
        )
        zero_points = create_constant_placeholder(
            exported_program,
            node.graph,
            node.name + "_zero_points",
            InputKind.BUFFER,
            torch.zeros(3, dtype=torch.int32),
            persistent_buffer=True,
        )

    replacement_args = (
        input_node,
        packed_weight,
        bias,
        scales,
        zero_points,
        list(stride),
        list(padding),
    )

    if is_pointwise:
        replacement_args += (linear,)
    elif not is_depthwise:
        scratch_bytes = 2 * PULP_NUM_CORES * in_channels * kernel_h * kernel_w
        replacement_args += (scratch_bytes,)

    return DialectNodeSpec(replacement_op, replacement_args)


@AtenToPulpPass.register_dialect_substitution(
    exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
)
def _get_quantize_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec:
    del dialect_pass
    if node.args[5] != torch.uint8:
        raise ValueError("PULP boundary quantization requires uint8 activations")
    return DialectNodeSpec(
        exir_ops.edge.pulp.quantize_per_tensor_cluster.default, node.args
    )


@AtenToPulpPass.register_dialect_substitution(
    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
)
def _get_dequantize_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec:
    del dialect_pass
    if node.args[5] != torch.uint8:
        raise ValueError("PULP boundary dequantization requires uint8 activations")
    return DialectNodeSpec(
        exir_ops.edge.pulp.dequantize_per_tensor_cluster.default, node.args
    )
