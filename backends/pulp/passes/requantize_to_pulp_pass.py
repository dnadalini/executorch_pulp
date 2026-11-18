# Copyright (c) 2026 KU Leuven.
"""Remove intermediate FP32 buffers from direct UINT8 DQ/Q pairs."""

from fractions import Fraction

import torch
from executorch.backends.pulp.ops.integer_ops import fixed_point
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from .integer_region_lowering import params_buffer


class RequantizeToPulpPass(ExportPass):
    def __init__(self, exported_program):
        super().__init__()
        self.exported_program = exported_program

    def call(self, graph_module):
        modified = False
        quantizers = {
            exir_ops.edge.pulp.quantize_per_tensor_cluster.default,
            exir_ops.edge.pulp.quantize_per_tensor.default,
        }
        dequantizers = {
            exir_ops.edge.pulp.dequantize_per_tensor_cluster.default,
            exir_ops.edge.pulp.dequantize_per_tensor.default,
        }
        for node in list(graph_module.graph.nodes):
            if node.target not in quantizers:
                continue
            source = node.args[0]
            if (
                source.target not in dequantizers
                or node.args[5] not in (torch.uint8, torch.int8)
                or source.args[5] != node.args[5]
            ):
                continue
            si, zi = source.args[1:3]
            so, zo, qmin, qmax = node.args[1:5]
            try:
                multipliers, shift = fixed_point(
                    [Fraction(float(si)) / Fraction(float(so))]
                )
            except (ValueError, OverflowError, ZeroDivisionError):
                continue
            params = params_buffer(
                node, self, [*multipliers, shift, zi, zo, qmin, qmax]
            )
            with node.graph.inserting_before(node):
                replacement = node.graph.call_function(
                    exir_ops.edge.pulp.requantize_per_tensor.default,
                    (source.args[0], params),
                )
                replacement.meta = node.meta.copy()
            node.replace_all_uses_with(replacement)
            node.graph.erase_node(node)
            modified = True
        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
