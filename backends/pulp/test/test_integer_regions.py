# Copyright (c) 2026 KU Leuven.
"""Check calibrated byte regions, shared residual branches and public boundaries."""

import unittest

import torch
from executorch.backends.pulp.edge_compile_config import pulp_edge_compile_config
from executorch.backends.pulp.export import prepare_for_pulp_export, pulp_export_context
from executorch.backends.pulp.passes.pulp_pass_manager import PulpPassManager
from executorch.backends.pulp.quantizer.quantizer import PulpQuantizer
from executorch.exir import to_edge
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class Residual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Conv2d(3, 8, 1)
        self.branch = torch.nn.Conv2d(8, 8, 1)
        self.classifier = torch.nn.Linear(8, 3)

    def forward(self, x):
        skip = self.first(x)
        branch = self.branch(skip)
        residual = branch + skip
        # Reuse both the skip and residual in different downstream consumers.
        result = residual + skip
        return self.classifier(result.mean((2, 3)))


class TestIntegerRegions(unittest.TestCase):
    def test_shared_residual_dynamic_pool_and_public_float_boundaries(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        data = torch.rand(1, 3, 4, 5).contiguous(memory_format=torch.channels_last)
        spec = (
            {
                2: torch.export.Dim("h", min=2, max=7),
                3: torch.export.Dim("w", min=2, max=9),
            },
        )
        with pulp_export_context():
            model = prepare_for_pulp_export(
                Residual().eval().to(memory_format=torch.channels_last)
            )
            prepared = prepare_pt2e(
                torch.export.export(model, (data,), dynamic_shapes=spec).module(),
                PulpQuantizer(),
            )
            prepared(data)
            edge = to_edge(
                torch.export.export(
                    convert_pt2e(prepared), (data,), dynamic_shapes=spec
                ),
                compile_config=pulp_edge_compile_config(),
            ).transform(PulpPassManager())
        ep = edge.exported_program()
        calls = [n for n in ep.graph.nodes if n.op == "call_function"]
        adds = [n for n in calls if "pulp.quantized_add" in str(n.target)]
        pools = [
            n for n in calls if "pulp.quantized_global_avg_pool2d" in str(n.target)
        ]
        quantizers = [
            n for n in calls if "pulp.quantize_per_tensor_cluster" in str(n.target)
        ]
        dequantizers = [
            n for n in calls if "pulp.dequantize_per_tensor_cluster" in str(n.target)
        ]
        self.assertEqual(
            (len(adds), len(pools), len(quantizers), len(dequantizers)), (2, 1, 1, 1)
        )
        for node in calls:
            value = node.meta.get("val")
            if isinstance(value, torch.Tensor) and node not in dequantizers:
                self.assertEqual(value.dtype, torch.uint8, str(node))
        self.assertGreater(len(adds[0].args[1].users), 1)
        for h, w in ((2, 9), (7, 2), (7, 9)):
            value = torch.rand(1, 3, h, w).contiguous(memory_format=torch.channels_last)
            result = ep.module()(value)
            self.assertEqual(
                (result.dtype, tuple(result.shape)), (torch.float32, (1, 3))
            )
        self.assertTrue(edge.to_executorch().buffer)

    def test_legacy_float_residual_option(self):
        data = torch.rand(1, 3, 4, 5).contiguous(memory_format=torch.channels_last)
        with pulp_export_context():
            model = prepare_for_pulp_export(
                Residual().eval().to(memory_format=torch.channels_last)
            )
            prepared = prepare_pt2e(
                torch.export.export(model, (data,)).module(),
                PulpQuantizer(quantize_residuals=False),
            )
            prepared(data)
            edge = to_edge(
                torch.export.export(convert_pt2e(prepared), (data,)),
                compile_config=pulp_edge_compile_config(),
            ).transform(PulpPassManager())
        targets = [str(n.target) for n in edge.exported_program().graph.nodes]
        self.assertEqual(sum("pulp.add_cluster" in t for t in targets), 2)
        self.assertFalse(any("pulp.quantized_add" in t for t in targets))
        self.assertEqual(edge.exported_program().module()(data).dtype, torch.float32)
        self.assertTrue(edge.to_executorch().buffer)

    def test_direct_dq_q_uses_integer_requantization_and_retains_shared_dq(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                floating = torch.ops.pulp.dequantize_per_tensor_cluster(
                    x, 0.125, 77, 0, 255, torch.uint8
                )
                quantized = torch.ops.pulp.quantize_per_tensor_cluster(
                    floating, 0.25, 33, 0, 255, torch.uint8
                )
                return quantized, floating

        data = torch.arange(256).byte()
        edge = to_edge(
            torch.export.export(Model(), (data,)),
            compile_config=pulp_edge_compile_config(),
        ).transform(PulpPassManager())
        targets = [str(n.target) for n in edge.exported_program().graph.nodes]
        self.assertEqual(sum("pulp.requantize_per_tensor" in t for t in targets), 1)
        self.assertEqual(
            sum("pulp.dequantize_per_tensor_cluster" in t for t in targets), 1
        )
        torch.testing.assert_close(
            edge.exported_program().module()(data), Model()(data), rtol=0, atol=0
        )
        self.assertTrue(edge.to_executorch().buffer)

    def test_extreme_residual_scale_retains_explicit_float_fallback(self):
        from executorch.backends.arm._passes.quant_args import QuantArgs
        from executorch.backends.pulp.passes.aten_to_pulp_pass import AtenToPulpPass

        class ByteRegion(torch.nn.Module):
            def forward(self, a, b):
                return torch.ops.pulp.add_cluster(a, b)

        a, b = torch.arange(256).byte(), torch.arange(256).byte()
        edge = to_edge(
            torch.export.export(ByteRegion(), (a, b)),
            compile_config=pulp_edge_compile_config(),
        )
        ep = edge.exported_program()
        node = next(n for n in ep.graph.nodes if "pulp.add_cluster" in str(n.target))
        node.meta["input_qparams"] = {
            0: QuantArgs(1.0, 7, 0, 255, torch.uint8),
            1: QuantArgs(1.0, 17, 0, 255, torch.uint8),
        }
        node.meta["output_qparams"] = {0: QuantArgs(1e-20, 0, 0, 255, torch.uint8)}
        result = AtenToPulpPass(ep)(ep.graph_module).graph_module
        targets = [str(n.target) for n in result.graph.nodes]
        self.assertFalse(any("pulp.quantized_add" in t for t in targets))
        self.assertEqual(
            sum("pulp.dequantize_per_tensor_cluster" in t for t in targets), 2
        )
        self.assertEqual(
            sum("pulp.quantize_per_tensor_cluster" in t for t in targets), 1
        )
        expected = (
            torch.round(((a.float() - 7) + (b.float() - 17)) / 1e-20)
            .clamp(0, 255)
            .byte()
        )
        torch.testing.assert_close(result(a, b)[0], expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
