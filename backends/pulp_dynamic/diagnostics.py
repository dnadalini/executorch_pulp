# Copyright (c) 2026 KU Leuven.
"""Read-only export diagnostics, with the same stage names as the PULP tests."""

import json
from pathlib import Path

import torch
from torch.fx import Node


def quantization_records(program):
    """Capture annotations and small serialized parameter buffers, not weights."""
    records = []
    buffers = program.graph_signature.inputs_to_buffers
    for node in program.graph.nodes:
        if node.op != "call_function":
            continue
        record = {"node": node.name, "operator": str(node.target)}
        for key in ("input_qparams", "output_qparams"):
            params = node.meta.get(key)
            if params:
                record[key] = {
                    str(index): value._asdict() for index, value in params.items()
                }
        target = str(node.target)
        if "pulp_dynamic.quantized_" in target:
            record["out_shift"] = node.args[5]
            record["layer_id"] = node.args[6]
            record["kind"] = node.args[8]
        elif (
            any(
                name in target
                for name in ("quantize_per_tensor", "dequantize_per_tensor")
            )
            and len(node.args) >= 6
        ):
            record["conversion"] = dict(
                zip(
                    ("scale", "zero_point", "quant_min", "quant_max", "dtype"),
                    node.args[1:6],
                )
            )
        for argument in node.args:
            if not isinstance(argument, Node):
                continue
            buffer_name = buffers.get(argument.name)
            value = program.state_dict.get(buffer_name)
            if (
                isinstance(value, torch.Tensor)
                and value.dtype == torch.int64
                and value.numel() <= 16
            ):
                record.setdefault("parameter_buffers", {})[buffer_name] = value.tolist()
        if len(record) > 2:
            records.append(record)
    return records


class ExportDiagnostics:
    """Write each lowering snapshot immediately, before later passes mutate it."""

    def __init__(self, logs):
        self.logs = Path(logs)
        self.pass_logs = self.logs / "lowering_passes"
        self.pass_logs.mkdir(parents=True, exist_ok=True)
        # Remove only snapshots owned by this logger from a previous export.
        for old in self.pass_logs.glob("[0-9][0-9]_*.info"):
            old.unlink()
        self.index = 0
        self.folded_qparams = "[]"

    def record_pass(self, name, program):
        """Write a full ExportedProgram and its quantization annotations."""
        params = json.dumps(quantization_records(program), indent=2, default=str)
        path = self.pass_logs / f"{self.index:02d}_{name}.info"
        path.write_text(str(program) + "\n\nQuantization parameters:\n" + params + "\n")
        self.index += 1
        if name == "FoldAndAnnotateQParamsPass":
            self.folded_qparams = params

    def write_quantization(self, program):
        """Retain folded scales even when subsequent passes remove annotations."""
        (self.logs / "quantization_parameters.info").write_text(
            json.dumps(
                {
                    "after_qdq_folding": json.loads(self.folded_qparams),
                    "after_lowering": quantization_records(program),
                },
                indent=2,
                default=str,
            )
            + "\n"
        )

    def write_runtime(self, program):
        """Match the existing test's serialized program and allocation dumps."""
        (self.logs / "04_executorch_program_buffer.info").write_text(
            str(program.buffer)
        )
        (self.logs / "04_executorch_program_exported_program.info").write_text(
            str(program.exported_program())
        )
        plan = program.executorch_program.execution_plan[0]
        (self.logs / "04_memory_plan.info").write_text(
            f"non_const_buffer_sizes={plan.non_const_buffer_sizes}\n"
            + "\n".join(
                f"value[{i}]: sizes={value.val.sizes} "
                f"shape_dynamism={value.val.shape_dynamism.name} "
                f"allocation={value.val.allocation_info}"
                for i, value in enumerate(plan.values)
                if hasattr(value.val, "shape_dynamism")
            )
            + "\n"
        )
        with (self.logs / "04_execution_plan.info").open("w") as stream:
            program.dump_executorch_program(verbose=True, out=stream)
