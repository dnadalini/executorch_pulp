# Copyright (c) 2026 KU Leuven.
"""Lower activations and absorb exclusive UINT8 conversion boundaries."""

import math

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node


class ActivationToPulpPass(ExportPass):
    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        q_op = exir_ops.edge.pulp.quantize_per_tensor_cluster.default
        dq_op = exir_ops.edge.pulp.dequantize_per_tensor_cluster.default
        for node in list(graph_module.graph.nodes):
            if node.op != "call_function":
                continue
            target = str(node.target)
            lower, upper = 0.0, math.inf
            if "aten.relu." in target:
                kind = 0
            elif "aten.relu6." in target:
                kind, upper = 1, 6.0
            elif "aten.hardtanh." in target:
                kind = 1
                lower = node.args[1] if len(node.args) > 1 else -1.0
                upper = node.args[2] if len(node.args) > 2 else 1.0
            elif "aten.gelu." in target:
                approximate = node.kwargs.get(
                    "approximate", node.args[1] if len(node.args) > 1 else "none"
                )
                if approximate not in ("none", "tanh"):
                    continue
                kind = 2 if approximate == "none" else 3
            else:
                continue
            value = node.meta.get("val")
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
                continue
            source = node.args[0]
            input_scale, input_zp = 1.0, 0
            # Keep shared dequantizers for their other consumers, but read the
            # byte source directly in this activation's own cluster invocation.
            if isinstance(source, Node) and source.target == dq_op:
                input_scale, input_zp = source.args[1:3]
                source = source.args[0]
            output_scale, output_zp, qmin, qmax = 1.0, 0, 0, 255
            output_node = node
            quantized_output = False
            if len(node.users) == 1:
                user = next(iter(node.users))
                if user.target == q_op and user.args[5] == torch.uint8:
                    output_node = user
                    output_scale, output_zp, qmin, qmax = user.args[1:5]
                    quantized_output = True
            args = (
                source,
                kind,
                float(lower),
                float(upper),
                input_scale,
                input_zp,
                output_scale,
                output_zp,
                qmin,
                qmax,
                quantized_output,
            )
            with graph_module.graph.inserting_before(node):
                replacement = graph_module.graph.call_function(
                    exir_ops.edge.pulp.activation_cluster.default, args
                )
                replacement.meta = output_node.meta.copy()
            output_node.replace_all_uses_with(replacement)
            graph_module.graph.erase_node(output_node)
            if output_node is not node:
                graph_module.graph.erase_node(node)
            modified = True
        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
