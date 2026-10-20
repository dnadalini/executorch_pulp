# Copyright (c) 2026 ETH Zurich and University of Bologna.
# All rights reserved.

import unittest

import torch
from executorch.backends.pulp.edge_compile_config import pulp_edge_compile_config
from executorch.backends.pulp.passes.pulp_pass_manager import PulpPassManager
from executorch.backends.pulp.quantizer.quantizer import PulpQuantizer
from executorch.exir import ExecutorchBackendConfig, to_edge
from executorch.exir.schema import TensorShapeDynamism
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class _Conv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 32, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TestPulpQuantizedConv2d(unittest.TestCase):
    def test_bounded_spatial_shapes_survive_lowering_and_serialization(self) -> None:
        torch.manual_seed(0)
        example = torch.rand(1, 32, 6, 7).to(memory_format=torch.channels_last)
        spatial = {
            2: torch.export.Dim("height", min=3, max=8),
            3: torch.export.Dim("width", min=3, max=8),
        }
        exported = torch.export.export(
            _Conv().eval(), (example,), dynamic_shapes=({**spatial},)
        )
        prepared = prepare_pt2e(exported.module(), PulpQuantizer())
        prepared(example)
        converted = convert_pt2e(prepared)
        edge = to_edge(
            torch.export.export(converted, (example,), dynamic_shapes=(spatial,)),
            compile_config=pulp_edge_compile_config(),
        ).transform(PulpPassManager())
        self.assertEqual(len(edge.exported_program().range_constraints), 2)
        reference = edge.exported_program().module()
        # Exercise sizes above the export example, shrink, grow and odd shapes.
        for h, w in ((8, 8), (3, 3), (3, 8), (8, 3), (5, 7), (8, 8)):
            raw = torch.randint(0, 256, (1, 32, h, w), dtype=torch.uint8)
            raw = raw.contiguous(memory_format=torch.channels_last)
            output = reference(raw)
            self.assertEqual(tuple(output.shape), (1, 32, h, w))
            self.assertTrue(output.is_contiguous(memory_format=torch.channels_last))
        with self.assertRaises(RuntimeError):
            reference(
                torch.zeros(1, 32, 2, 8, dtype=torch.uint8).contiguous(
                    memory_format=torch.channels_last
                )
            )

        program = edge.to_executorch(ExecutorchBackendConfig())
        plan = program.executorch_program.execution_plan[0]
        for value_index in (plan.inputs[0], plan.outputs[0]):
            tensor = plan.values[value_index].val
            self.assertEqual(tensor.shape_dynamism, TensorShapeDynamism.DYNAMIC_BOUND)
            self.assertEqual(tensor.sizes, [1, 32, 8, 8])
        self.assertTrue(program.buffer)

    def test_out_reference_resizes_between_invocations(self) -> None:
        weight = torch.ones(32, 3, 3, 8, dtype=torch.int8)
        out = torch.empty(
            1, 32, 8, 8, dtype=torch.uint8, memory_format=torch.channels_last
        )
        for h, w in ((3, 5), (8, 8), (7, 3)):
            raw = torch.ones(1, 8, h, w, dtype=torch.uint8).contiguous(
                memory_format=torch.channels_last
            )
            expected = torch.ops.pulp.quantized_conv2d(
                raw, weight, [1, 1], [1, 1], 1, 1152
            )
            torch.ops.pulp.quantized_conv2d.out(
                raw, weight, [1, 1], [1, 1], 1, 1152, out=out
            )
            self.assertEqual(tuple(out.shape), (1, 32, h, w))
            torch.testing.assert_close(out, expected)
            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))

    def test_quantize_lower_and_execute_reference(self) -> None:
        torch.manual_seed(0)
        model = _Conv().eval()
        example = torch.rand(1, 32, 8, 8).to(memory_format=torch.channels_last)

        exported = torch.export.export(model, (example,))
        prepared = prepare_pt2e(exported.module(), PulpQuantizer())
        prepared(example)
        converted = convert_pt2e(prepared)
        edge = to_edge(
            torch.export.export(converted, (example,)),
            compile_config=pulp_edge_compile_config(),
        ).transform(PulpPassManager())

        graph = edge.exported_program().graph
        pulp_nodes = [
            node for node in graph.nodes if "pulp.quantized_conv2d" in str(node.target)
        ]
        self.assertEqual(len(pulp_nodes), 1)
        self.assertEqual(pulp_nodes[0].args[-1], 2 * 8 * 32 * 3 * 3)
        self.assertFalse(
            any("aten.convolution" in str(node.target) for node in graph.nodes)
        )

        placeholders = [node for node in graph.nodes if node.op == "placeholder"]
        self.assertEqual(len(placeholders), 2)
        raw_input = torch.arange(32 * 8 * 8, dtype=torch.int32).remainder(256)
        raw_input = raw_input.to(torch.uint8).reshape(1, 32, 8, 8)
        raw_input = raw_input.contiguous(memory_format=torch.channels_last)
        output = edge.exported_program().module()(raw_input)
        self.assertEqual(output.dtype, torch.uint8)
        self.assertEqual(tuple(output.shape), (1, 32, 8, 8))
        self.assertTrue(output.is_contiguous(memory_format=torch.channels_last))


if __name__ == "__main__":
    unittest.main()
