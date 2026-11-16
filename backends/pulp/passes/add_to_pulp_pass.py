# Copyright (c) 2026 KU Leuven.
"""Lower same-shaped FP32 additions and remove redundant residual layout copies."""

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node
from torch.fx.experimental.symbolic_shapes import statically_known_true


def _value(node):
    value = node.meta.get("val") if isinstance(node, Node) else None
    return value if isinstance(value, torch.Tensor) else None


def _compatible(left, right, output):
    values = [_value(node) for node in (left, right, output)]
    if any(value is None or value.dtype != torch.float32 for value in values):
        return False
    a, b, out = values
    if a.dim() != b.dim() or a.dim() != out.dim():
        return False
    if not all(statically_known_true(x == y) for x, y in zip(a.shape, b.shape)):
        return False
    return a.dim_order() == b.dim_order() == out.dim_order()


def _unwrap_copy(node):
    if (
        isinstance(node, Node)
        and node.target == exir_ops.edge.dim_order_ops._to_dim_order_copy.default
    ):
        original = node.args[0]
        before, after = _value(original), _value(node)
        if (
            before is not None
            and after is not None
            and before.dtype == after.dtype == torch.float32
        ):
            return original
    return node


class AddToPulpPass(ExportPass):
    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        copy_op = exir_ops.edge.dim_order_ops._to_dim_order_copy.default
        for node in list(graph_module.graph.nodes):
            if node.target != exir_ops.edge.aten.add.Tensor:
                continue
            alpha = node.kwargs.get("alpha", node.args[2] if len(node.args) > 2 else 1)
            if not isinstance(alpha, (int, float)):
                continue
            left, right = node.args[:2]
            original_left, original_right = _unwrap_copy(left), _unwrap_copy(right)
            output = node
            if len(node.users) == 1:
                user = next(iter(node.users))
                if user.target == copy_op and _compatible(
                    original_left, original_right, user
                ):
                    output = user
            if _compatible(original_left, original_right, output):
                left, right = original_left, original_right
            elif not _compatible(left, right, output):
                continue
            with graph_module.graph.inserting_before(node):
                replacement = graph_module.graph.call_function(
                    exir_ops.edge.pulp.add_cluster.default, (left, right, float(alpha))
                )
                replacement.meta = output.meta.copy()
            output.replace_all_uses_with(replacement)
            graph_module.graph.erase_node(output)
            if output is not node:
                graph_module.graph.erase_node(node)
            modified = True
        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
