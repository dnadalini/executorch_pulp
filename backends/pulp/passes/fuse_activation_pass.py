# Copyright (c) 2026 KU Leuven.
"""Fuse only activation patterns explicitly accepted by the PULP quantizer."""

from executorch.backends.arm.common.annotation_meta import ArmAnnotationInfo
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node


class FuseConvActivationPass(ExportPass):
    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for activation in list(graph_module.graph.nodes):
            target = str(activation.target)
            if not any(f"aten.{op}." in target for op in ("relu", "relu6", "hardtanh")):
                continue
            conv = activation.args[0]
            if not isinstance(conv, Node) or len(conv.users) != 1:
                continue
            if conv.target not in (
                exir_ops.edge.aten.convolution.default,
                exir_ops.edge.pulp.conv2d_export.default,
                exir_ops.edge.aten.linear.default,
            ):
                continue
            # In an annotated composite pattern there is no Q/DQ between these
            # nodes, and the output observer belongs to the activation.
            custom = activation.meta.get("custom", {}).get(
                ArmAnnotationInfo.CUSTOM_META_KEY
            )
            if custom is None or not ArmAnnotationInfo(custom).quantized:
                continue
            if not activation.users or any(
                user.target
                != exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
                for user in activation.users
            ):
                continue
            if "aten.relu6." in target:
                bound = 6.0
            elif "aten.hardtanh." in target:
                lower = activation.args[1] if len(activation.args) > 1 else -1.0
                bound = activation.args[2] if len(activation.args) > 2 else 1.0
                if lower != 0 or bound <= 0:
                    continue
            else:
                bound = None
            conv.meta["pulp_activation_upper"] = bound
            activation.replace_all_uses_with(conv)
            graph_module.graph.erase_node(activation)
            modified = True
        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)
