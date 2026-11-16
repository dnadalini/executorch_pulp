# Copyright (c) 2026 KU Leuven.
"""Expose residual adds and global pooling before PT2E annotation."""

import math

import torch
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from executorch.backends.pulp.quantizer.quantization_config import (
    PULP_UINT8_ACTIVATION_QSPEC,
)
from torch.fx import Node
from torch.fx.experimental.symbolic_shapes import statically_known_true


REGIONS = {torch.ops.pulp.add_cluster.default, torch.ops.pulp.global_avg_pool2d.default}
REGION_CONFIG = QuantizationConfig(
    input_activation=PULP_UINT8_ACTIVATION_QSPEC,
    output_activation=PULP_UINT8_ACTIVATION_QSPEC,
    weight=None,
    bias=None,
    label="PULP UINT8 residual and pooling",
)


def _tensor(node):
    value = node.meta.get("val") if isinstance(node, Node) else None
    return (
        value
        if isinstance(value, torch.Tensor) and value.dtype == torch.float32
        else None
    )


def _copy_source(node):
    if isinstance(node, Node) and node.target == torch.ops.aten._to_copy.default:
        if _tensor(node.args[0]) is not None and _tensor(node) is not None:
            return node.args[0]
    return node


def promote_regions(model):
    graph = model.graph
    for node in list(graph.nodes):
        target = node.target
        if target in REGIONS:
            continue
        if target == torch.ops.aten.add.Tensor:
            a, b = node.args[:2]
            alpha = node.kwargs.get("alpha", 1)
            a, b = _copy_source(a), _copy_source(b)
            va, vb = _tensor(a), _tensor(b)
            if (
                va is None
                or vb is None
                or not isinstance(alpha, (int, float))
                or not math.isfinite(alpha)
            ):
                continue
            if (
                va.dim() != vb.dim()
                or not all(
                    statically_known_true(x == y) for x, y in zip(va.shape, vb.shape)
                )
                or va.dim_order() != vb.dim_order()
            ):
                continue
            output = node
            if len(node.users) == 1:
                user = next(iter(node.users))
                if (
                    user.target == torch.ops.aten._to_copy.default
                    and _tensor(user) is not None
                    and _tensor(user).dim_order() == va.dim_order()
                ):
                    output = user
            op, args = torch.ops.pulp.add_cluster.default, (a, b, float(alpha))
            value = torch.empty_like(va)
        elif target in (
            torch.ops.aten.mean.dim,
            torch.ops.aten.adaptive_avg_pool2d.default,
        ):
            source = node.args[0]
            value = _tensor(source)
            if (
                value is None
                or value.dim() != 4
                or not (
                    value.is_contiguous()
                    or value.is_contiguous(memory_format=torch.channels_last)
                )
            ):
                continue
            if target == torch.ops.aten.mean.dim:
                dims = node.args[1]
                if (
                    dims is None
                    or len(dims) != 2
                    or {d % 4 for d in dims} != {2, 3}
                    or node.kwargs.get("dtype") not in (None, torch.float32)
                ):
                    continue
                keepdim = (
                    node.args[2]
                    if len(node.args) > 2
                    else node.kwargs.get("keepdim", False)
                )
            else:
                if tuple(node.args[1]) != (1, 1):
                    continue
                keepdim = True
            output = node
            op, args = torch.ops.pulp.global_avg_pool2d.default, (source, keepdim)
            value = node.meta["val"]
        else:
            continue
        with graph.inserting_before(node):
            replacement = graph.call_function(op, args)
            replacement.meta = output.meta.copy()
            replacement.meta["val"] = value
        output.replace_all_uses_with(replacement)
        graph.erase_node(output)
        if output is not node:
            graph.erase_node(node)
    graph.eliminate_dead_code()
    model.recompile()
    return model
