# Copyright (c) 2026 KU Leuven.
"""Lower rank-four FP32 spatial means to channel-tiled cluster pooling."""

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule


class GlobalPoolToPulpPass(ExportPass):
    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for node in list(graph_module.graph.nodes):
            if node.target == exir_ops.edge.aten.mean.dim:
                dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
                if (
                    dims is None
                    or len(dims) != 2
                    or any(d < -4 or d >= 4 for d in dims)
                    or {d % 4 for d in dims} != {2, 3}
                ):
                    continue
                keepdim = (
                    node.args[2]
                    if len(node.args) > 2
                    else node.kwargs.get("keepdim", False)
                )
                if node.kwargs.get("dtype") not in (None, torch.float32):
                    continue
            elif node.target == exir_ops.edge.aten.adaptive_avg_pool2d.default:
                if tuple(node.args[1]) != (1, 1):
                    continue
                keepdim = True
            else:
                continue
            source = node.args[0]
            value = source.meta.get("val")
            if (
                not isinstance(value, torch.Tensor)
                or value.dim() != 4
                or value.dtype != torch.float32
            ):
                continue
            if not (
                value.is_contiguous()
                or value.is_contiguous(memory_format=torch.channels_last)
            ):
                continue
            if value.shape[2] == 0 or value.shape[3] == 0:
                continue
            copy_op = exir_ops.edge.dim_order_ops._to_dim_order_copy.default
            if source.target == copy_op:
                original = source.args[0]
                original_value = original.meta.get("val")
                if (
                    isinstance(original_value, torch.Tensor)
                    and original_value.dtype == torch.float32
                    and (
                        original_value.is_contiguous()
                        or original_value.is_contiguous(
                            memory_format=torch.channels_last
                        )
                    )
                ):
                    source = original
            with graph_module.graph.inserting_before(node):
                replacement = graph_module.graph.call_function(
                    exir_ops.edge.pulp.global_avg_pool2d.default, (source, keepdim)
                )
                replacement.meta = node.meta.copy()
            node.replace_all_uses_with(replacement)
            graph_module.graph.erase_node(node)
            modified = True
        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
