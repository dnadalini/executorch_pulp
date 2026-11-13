# Copyright (c) 2026 KU Leuven.
"""Regression for dynamic convolutions whose spatial dimensions cross one."""

import unittest

import torch
from executorch.backends.pulp.edge_compile_config import pulp_edge_compile_config
from executorch.backends.pulp.export import prepare_for_pulp_export, pulp_export_context
from executorch.backends.pulp.passes.pulp_pass_manager import PulpPassManager
from executorch.backends.pulp.quantizer.quantizer import PulpQuantizer
from executorch.exir import to_edge
from torch.fx.experimental import _config as symbolic_config
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class TestDynamicExport(unittest.TestCase):
    def test_layout_boundary_survives_export_quantization_and_serialization(self):
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = torch.nn.Conv2d(3, 8, 3, stride=16, padding=1)
                self.pointwise = torch.nn.Conv2d(8, 8, 1)
                self.depthwise = torch.nn.Conv2d(8, 8, 3, padding=1, groups=8)
                self.relu = torch.nn.ReLU6()
                self.classifier = torch.nn.Linear(8, 3)

            def forward(self, x):
                x = self.relu(self.stem(x))
                x = self.relu(self.pointwise(x))
                x = self.relu(self.depthwise(x))
                return self.classifier(x.mean((2, 3)))

        torch.manual_seed(37)
        torch.set_num_threads(1)
        shapes = ((32, 32), (16, 16), (16, 32), (32, 16), (17, 31), (23, 19))
        samples = [
            torch.rand(1, 3, h, w).contiguous(memory_format=torch.channels_last)
            for h, w in shapes
        ]
        model = Network().eval().to(memory_format=torch.channels_last)
        expected = [model(sample) for sample in samples]
        weight = model.stem.weight
        state_before = symbolic_config.backed_size_oblivious
        with pulp_export_context():
            model = prepare_for_pulp_export(model)
            self.assertIs(model.stem.weight, weight)
            spatial = {
                2: torch.export.Dim("h", min=16, max=32),
                3: torch.export.Dim("w", min=16, max=32),
            }
            exported = torch.export.export(
                model, (samples[0],), dynamic_shapes=(spatial,)
            )
            for sample, golden in zip(samples, expected):
                torch.testing.assert_close(exported.module()(sample), golden)
            prepared = prepare_pt2e(exported.module(), PulpQuantizer())
            for sample in samples:
                prepared(sample)
            converted = torch.export.export(
                convert_pt2e(prepared), (samples[0],), dynamic_shapes=(spatial,)
            )
            edge = to_edge(
                converted, compile_config=pulp_edge_compile_config()
            ).transform(PulpPassManager())
            graph = edge.exported_program().graph
            self.assertFalse(
                any("pulp.conv2d_export" in str(node.target) for node in graph.nodes)
            )
            for kind in (
                "quantized_conv2d_clamp",
                "quantized_pointwise_clamp",
                "quantized_depthwise_conv2d_clamp",
            ):
                self.assertTrue(any(kind in str(node.target) for node in graph.nodes))
            for sample in samples:
                result = edge.exported_program().module()(sample)
                self.assertEqual(result.shape, (1, 3))
                self.assertTrue(torch.isfinite(result).all())
            with self.assertRaises(RuntimeError):
                exported.module()(
                    torch.rand(1, 3, 15, 32).contiguous(
                        memory_format=torch.channels_last
                    )
                )
            program = edge.to_executorch()
            self.assertTrue(program.buffer)
            self.assertFalse(
                any(
                    "conv2d_export" in op.name
                    for op in program.executorch_program.execution_plan[0].operators
                )
            )
        self.assertEqual(symbolic_config.backed_size_oblivious, state_before)

    def test_export_only_operator_is_removed_without_quantization(self):
        data = torch.rand(1, 3, 4, 5).contiguous(memory_format=torch.channels_last)
        original = torch.nn.Conv2d(3, 4, 1).eval()
        wrapped = prepare_for_pulp_export(original)
        torch.testing.assert_close(wrapped(data), original(data))
        self.assertIs(wrapped.weight, original.weight)
        with pulp_export_context():
            edge = to_edge(
                torch.export.export(wrapped, (data,)),
                compile_config=pulp_edge_compile_config(),
            ).transform(PulpPassManager())
        self.assertFalse(
            any(
                "pulp.conv2d_export" in str(node.target)
                for node in edge.exported_program().graph.nodes
            )
        )
        torch.testing.assert_close(
            edge.exported_program().module()(data), original(data)
        )


if __name__ == "__main__":
    unittest.main()
