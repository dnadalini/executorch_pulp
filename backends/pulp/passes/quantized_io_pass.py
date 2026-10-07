# Copyright (c) 2026 ETH Zurich and University of Bologna.
# All rights reserved.

from typing import cast

import torch
from executorch.backends.arm.constants import DQ_OPS, Q_OPS
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node


class PulpQuantizedIOPass(ExportPass):
    """Expose the boundary uint8 tensors directly as method inputs/outputs."""

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for node in list(graph_module.graph.nodes):
            if node.target in Q_OPS and isinstance(node.args[0], Node):
                input_node = cast(Node, node.args[0])
                if input_node.op == "placeholder":
                    input_node.meta["val"] = node.meta["val"]
                    node.replace_all_uses_with(input_node)
                    graph_module.graph.erase_node(node)
                    modified = True
            elif node.target in DQ_OPS and all(
                user.op == "output" for user in node.users
            ):
                quantized_value = cast(Node, node.args[0])
                node.replace_all_uses_with(quantized_value)
                graph_module.graph.erase_node(node)
                modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)

