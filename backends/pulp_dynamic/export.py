# Copyright (c) 2026 KU Leuven.
"""Fold graph-adjacent eval batchnorm, then reuse channels-last export."""

import copy

import torch
from executorch.backends.pulp.export import prepare_for_pulp_export, pulp_export_context

pulp_dynamic_export_context = pulp_export_context


def prepare_for_pulp_dynamic_export(model):
    model = copy.deepcopy(model).eval()
    # Trace before wrapping Conv2d, so graph connectivity decides fold eligibility.
    graph = torch.fx.symbolic_trace(model)
    for node in list(graph.graph.nodes):
        if node.op != "call_module":
            continue
        bn = graph.get_submodule(node.target)
        if not isinstance(bn, torch.nn.BatchNorm2d) or not bn.track_running_stats:
            continue
        source = node.args[0]
        if source.op != "call_module" or len(source.users) != 1:
            continue
        conv = graph.get_submodule(source.target)
        if not isinstance(conv, torch.nn.Conv2d):
            continue
        graph.set_submodule(
            source.target, torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
        )
        node.replace_all_uses_with(source)
        graph.graph.erase_node(node)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    return prepare_for_pulp_export(graph)
