# Copyright (c) 2026 KU Leuven.
"""Satisfy the portable permute kernel's matching-dimension-order contract."""

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class PortableLayoutPass(ExportPass):
    def call(self, graph_module):
        changed = False
        for node in list(graph_module.graph.nodes):
            if node.target != exir_ops.edge.aten.permute_copy.default:
                continue
            source = node.args[0]
            before, after = source.meta.get("val"), node.meta.get("val")
            if not isinstance(before, torch.Tensor) or not isinstance(
                after, torch.Tensor
            ):
                continue
            if before.dim_order() == after.dim_order():
                continue
            with node.graph.inserting_before(node):
                copy = node.graph.call_function(
                    exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                    (source,),
                    {"dim_order": list(after.dim_order())},
                )
            node.args = (copy, *node.args[1:])
            changed = True
        if changed:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, changed)
