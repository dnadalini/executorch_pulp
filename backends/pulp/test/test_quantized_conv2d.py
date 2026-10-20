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
    def __init__(self, bias: bool = True) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 32, 3, padding=1, bias=bias)

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
                raw,
                weight,
                None,
                torch.tensor([1.0, 1.0, 2.0]),
                torch.zeros(3, dtype=torch.int32),
                [1, 1],
                [1, 1],
                1152,
            )
            torch.ops.pulp.quantized_conv2d.out(
                raw,
                weight,
                None,
                torch.tensor([1.0, 1.0, 2.0]),
                torch.zeros(3, dtype=torch.int32),
                [1, 1],
                [1, 1],
                1152,
                out=out,
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
        self.assertEqual(len(placeholders), 5)
        conv = pulp_nodes[0]
        state = edge.exported_program().state_dict
        targets = edge.exported_program().graph_signature.inputs_to_buffers
        bias = state[targets[conv.args[2].name]]
        scales = state[targets[conv.args[3].name]]
        zero_points = state[targets[conv.args[4].name]]
        self.assertEqual(bias.dtype, torch.int32)
        self.assertGreater(bias.abs().max().item(), 127)
        self.assertEqual(scales.shape, (3,))
        self.assertTrue((scales > 0).all())
        self.assertTrue((zero_points == 0).all())
        expected_bias = torch.round(
            model.conv.bias.detach() / (scales[0] * scales[1])
        ).to(torch.int32)
        torch.testing.assert_close(bias, expected_bias)
        raw_input = torch.arange(32 * 8 * 8, dtype=torch.int32).remainder(256)
        raw_input = raw_input.to(torch.uint8).reshape(1, 32, 8, 8)
        raw_input = raw_input.contiguous(memory_format=torch.channels_last)
        output = edge.exported_program().module()(raw_input)
        self.assertEqual(output.dtype, torch.uint8)
        self.assertEqual(tuple(output.shape), (1, 32, 8, 8))
        self.assertTrue(output.is_contiguous(memory_format=torch.channels_last))

        # Independently compute the signed integer dot products and bias before
        # the right shift. Exercise negative weights/bias as well as saturation.
        weight = state[targets[conv.args[1].name]].permute(0, 3, 1, 2).to(torch.int64)
        patches = torch.nn.functional.unfold(
            raw_input.to(torch.float64), 3, padding=1
        ).to(torch.int64)
        accumulators = weight.flatten(1) @ patches[0] + bias.to(torch.int64)[:, None]
        shift = round(-torch.log2(scales[0] * scales[1] / scales[2]).item())
        expected = (
            (accumulators >> shift).clamp(0, 255).to(torch.uint8).reshape_as(output)
        )
        torch.testing.assert_close(output, expected)

    def test_bias_precision_saturation_and_live_scales(self) -> None:
        raw = torch.ones(1, 1, 3, 5, dtype=torch.uint8).contiguous(
            memory_format=torch.channels_last
        )
        weight = torch.ones(7, 1, 1, 1, dtype=torch.int8)
        bias = torch.tensor(
            [-100000, -2, 126, 254, 100000, 16777214, 16777215], dtype=torch.int32
        )
        zp = torch.zeros(3, dtype=torch.int32)
        for shift in (0, 1, 16, 17, 31):
            out = torch.ops.pulp.quantized_conv2d(
                raw,
                weight,
                bias,
                torch.tensor([1.0, 1.0, float(2**shift)]),
                zp,
                [1, 1],
                [0, 0],
                16,
            )
            expected = (
                ((bias.to(torch.int64) + 1) >> shift).clamp(0, 255).to(torch.uint8)
            )
            torch.testing.assert_close(
                out, expected[None, :, None, None].expand_as(out)
            )
        for scales, zero_points in (
            (torch.tensor([1.0, 1.0, 3.0]), zp),
            (torch.tensor([float("nan"), 1.0, 2.0]), zp),
            (torch.tensor([1.0, 1.0, 2.0]), torch.tensor([1, 0, 0], dtype=torch.int32)),
        ):
            with self.assertRaises(ValueError):
                torch.ops.pulp.quantized_conv2d(
                    raw, weight, bias, scales, zero_points, [1, 1], [0, 0], 16
                )

    def test_bias_free_model_still_lowers(self) -> None:
        example = torch.rand(1, 32, 4, 4).to(memory_format=torch.channels_last)
        exported = torch.export.export(_Conv(bias=False).eval(), (example,))
        prepared = prepare_pt2e(exported.module(), PulpQuantizer())
        prepared(example)
        converted = convert_pt2e(prepared)
        edge = to_edge(
            torch.export.export(converted, (example,)),
            compile_config=pulp_edge_compile_config(),
        ).transform(PulpPassManager())
        conv = next(
            n
            for n in edge.exported_program().graph.nodes
            if "pulp.quantized_conv2d" in str(n.target)
        )
        self.assertIsNone(conv.args[2])
        self.assertTrue(edge.to_executorch().buffer)

    def test_unrepresentable_bias_is_rejected(self) -> None:
        example = torch.ones(1, 32, 4, 4).to(memory_format=torch.channels_last)
        for value in (-1e5, 1e5):
            model = _Conv().eval()
            model.conv.weight.data.fill_(0.05)
            model.conv.bias.data.fill_(value)
            exported = torch.export.export(model, (example,))
            prepared = prepare_pt2e(exported.module(), PulpQuantizer())
            prepared(example)
            converted = convert_pt2e(prepared)
            edge = to_edge(
                torch.export.export(converted, (example,)),
                compile_config=pulp_edge_compile_config(),
            )
            with self.assertRaises(Exception) as raised:
                edge.transform(PulpPassManager())
            error = raised.exception
            while error.__cause__ is not None:
                error = error.__cause__
            self.assertIsInstance(error, ValueError)
            self.assertIn("int32 quantization range", str(error))


if __name__ == "__main__":
    unittest.main()
