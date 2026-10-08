# Copyright (c) 2026 ETH Zurich and University of Bologna.
# All rights reserved.

import inspect
from typing import Any, Type

from executorch.backends.arm._passes import FoldAndAnnotateQParamsPass
from executorch.backends.transforms.utils import delete_constant_placeholder
from executorch.exir.pass_base import (
    ExportedProgramPassBase,
    ExportedProgramPassResult,
    ExportPass,
)
from executorch.exir.pass_manager import ExportedProgramPassManager
from executorch.exir.program._program import _transform, lift_constant_tensor_pass
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind

from .aten_to_pulp_pass import AtenToPulpPass
from .quantized_io_pass import PulpQuantizedIOPass


PassClass = Type[ExportPass]


class _PulpLoweringPass(ExportedProgramPassBase):
    pass_classes: list[PassClass] = [
        FoldAndAnnotateQParamsPass,
        AtenToPulpPass,
        PulpQuantizedIOPass,
    ]

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        modified = False
        for pass_class in self.pass_classes:
            kwargs: dict[str, Any] = {}
            if "exported_program" in inspect.signature(pass_class).parameters:
                kwargs["exported_program"] = exported_program
            transformed = _transform(exported_program, pass_class(**kwargs))
            modified |= transformed is not exported_program
            exported_program = transformed

        dead_constants = []
        input_specs = {
            spec.arg.name: spec for spec in exported_program.graph_signature.input_specs
        }
        for node in exported_program.graph.nodes:
            spec = input_specs.get(node.name)
            if (
                node.op == "placeholder"
                and not node.users
                and spec is not None
                and spec.kind != InputKind.USER_INPUT
            ):
                dead_constants.append(node)
        for node in dead_constants:
            delete_constant_placeholder(exported_program, node)
        if dead_constants:
            exported_program.graph_module.recompile()
            modified = True

        buffer_count = len(exported_program.graph_signature.buffers)
        exported_program = lift_constant_tensor_pass(exported_program)
        modified |= len(exported_program.graph_signature.buffers) != buffer_count
        return ExportedProgramPassResult(exported_program, modified)


class PulpPassManager(ExportedProgramPassManager):
    def __init__(self) -> None:
        super().__init__([_PulpLoweringPass()])
