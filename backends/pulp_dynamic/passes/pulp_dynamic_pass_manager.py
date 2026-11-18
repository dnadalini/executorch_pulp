# Copyright (c) 2026 KU Leuven.
from executorch.backends.arm._passes import FoldAndAnnotateQParamsPass
from executorch.backends.pulp.passes.activation_to_pulp_pass import ActivationToPulpPass
from executorch.backends.pulp.passes.add_to_pulp_pass import AddToPulpPass
from executorch.backends.pulp.passes.fuse_activation_pass import FuseConvActivationPass
from executorch.backends.pulp.passes.global_pool_to_pulp_pass import (
    GlobalPoolToPulpPass,
)
from executorch.backends.pulp.passes.pulp_pass_manager import _PulpLoweringPass
from executorch.backends.pulp.passes.requantize_to_pulp_pass import RequantizeToPulpPass
from executorch.backends.pulp_dynamic.config import PulpDynamicConfig
from executorch.exir.pass_base import ExportedProgramPassResult
from executorch.exir.pass_manager import ExportedProgramPassManager

from .lowering import AtenToPulpDynamicPass
from .portable_layout import PortableLayoutPass


class DynamicLoweringPass(_PulpLoweringPass):
    def __init__(self, config, pass_snapshot_callback=None):
        super().__init__()
        self.config = config
        self.pass_snapshot_callback = pass_snapshot_callback

        class Configured(AtenToPulpDynamicPass):
            _unused = None

            def __init__(self, exported_program):
                super().__init__(exported_program, config)

        # Substitution registries are intentionally isolated between subclasses.
        Configured._DIALECT_SUBSTITUTIONS = (
            AtenToPulpDynamicPass._DIALECT_SUBSTITUTIONS.copy()
        )
        Configured.__name__ = "AtenToPulpDynamicPass"
        self.pass_classes = [
            FuseConvActivationPass,
            FoldAndAnnotateQParamsPass,
            Configured,
            ActivationToPulpPass,
            GlobalPoolToPulpPass,
            AddToPulpPass,
            RequantizeToPulpPass,
            PortableLayoutPass,
        ]

    def call(self, exported_program):
        result = super().call(exported_program)
        ep = result.exported_program
        for index, node in enumerate(
            n for n in ep.graph.nodes if n.op == "call_function"
        ):
            node.meta.setdefault("debug_handle", index)
            if "pulp_dynamic.quantized_" in str(node.target):
                args = list(node.args)
                args[6] = index
                node.args = tuple(args)
        ep.graph_module.recompile()
        if self.pass_snapshot_callback is not None:
            self.pass_snapshot_callback("AssignLayerIndices", ep)
        return ExportedProgramPassResult(ep, True)


class PulpDynamicPassManager(ExportedProgramPassManager):
    def __init__(self, config=None, *, pass_snapshot_callback=None):
        super().__init__(
            [DynamicLoweringPass(config or PulpDynamicConfig(), pass_snapshot_callback)]
        )
