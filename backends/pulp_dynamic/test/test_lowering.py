# Copyright (c) 2026 KU Leuven.
import unittest

import torch
from executorch.backends.pulp.edge_compile_config import pulp_edge_compile_config
from executorch.backends.pulp.export import prepare_for_pulp_export, pulp_export_context
from executorch.backends.pulp_dynamic.config import PulpDynamicConfig
from executorch.backends.pulp_dynamic.export import prepare_for_pulp_dynamic_export
from executorch.backends.pulp_dynamic.passes.pulp_dynamic_pass_manager import (
    PulpDynamicPassManager,
)
from executorch.backends.pulp_dynamic.quantizer.quantizer import PulpDynamicQuantizer
from executorch.exir import to_edge
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class Residual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.dw = torch.nn.Conv2d(8, 8, 3, padding=1, groups=8)
        self.pw = torch.nn.Conv2d(8, 8, 1)
        self.fc = torch.nn.Linear(8, 4)

    def forward(self, x):
        x = torch.nn.functional.relu6(self.conv(x))
        y = self.pw(torch.relu(self.dw(x)))
        return self.fc((x + y).mean((2, 3)))


class TestDynamicLowering(unittest.TestCase):
    def lower(self, model, data, spec=None, config=None):
        torch.set_num_threads(1)
        with pulp_export_context():
            model = prepare_for_pulp_export(model.eval())
            p = prepare_pt2e(
                torch.export.export(model, (data,), dynamic_shapes=spec).module(),
                PulpDynamicQuantizer(config),
            )
            p(data)
            edge = to_edge(
                torch.export.export(convert_pt2e(p), (data,), dynamic_shapes=spec),
                compile_config=pulp_edge_compile_config(),
            ).transform(PulpDynamicPassManager(config))
        return edge

    def test_dynamic_residual_activation_pool_classifier(self):
        torch.manual_seed(20)
        data = torch.randn(1, 3, 8, 7).contiguous(memory_format=torch.channels_last)
        spec = (
            {
                2: torch.export.Dim("h", min=3, max=16),
                3: torch.export.Dim("w", min=3, max=16),
            },
        )
        edge = self.lower(Residual().to(memory_format=torch.channels_last), data, spec)
        ep = edge.exported_program()
        targets = [str(n.target) for n in ep.graph.nodes]
        self.assertEqual(sum("pulp_dynamic.quantized_" in t for t in targets), 4)
        self.assertEqual(sum("pulp.activation_cluster" in t for t in targets), 0)
        self.assertEqual(sum("pulp.quantized_add" in t for t in targets), 1)
        self.assertEqual(
            sum("pulp.quantized_global_avg_pool2d" in t for t in targets), 1
        )
        for h, w in ((3, 16), (16, 3), (8, 7)):
            out = ep.module()(
                torch.randn(1, 3, h, w).contiguous(memory_format=torch.channels_last)
            )
            self.assertEqual((out.dtype, tuple(out.shape)), (torch.float32, (1, 4)))
        self.assertTrue(edge.to_executorch().buffer)

    def test_linear_multiple_rows(self):
        data = torch.randn(2, 3, 7)
        edge = self.lower(torch.nn.Linear(7, 11), data)
        self.assertEqual(edge.exported_program().module()(data).shape, (2, 3, 11))
        self.assertTrue(edge.to_executorch().buffer)

    def test_fused_activations(self):
        for kind in range(3):
            for activation in (
                torch.nn.ReLU(),
                torch.nn.ReLU6(),
                torch.nn.Hardtanh(0, 2),
            ):
                with self.subTest(kind=kind, activation=activation):
                    layer = torch.nn.Conv2d(
                        3, 3, 3 if kind != 1 else 1, groups=3 if kind == 2 else 1
                    )
                    with torch.no_grad():
                        layer.bias.copy_(torch.tensor([-20.0, 0.0, 20.0]))
                    model = torch.nn.Sequential(layer, activation).to(
                        memory_format=torch.channels_last
                    )
                    data = torch.randn(1, 3, 5, 5).contiguous(
                        memory_format=torch.channels_last
                    )
                    ep = self.lower(model, data).exported_program()
                    nodes = [
                        n
                        for n in ep.graph.nodes
                        if "pulp_dynamic.quantized_" in str(n.target)
                    ]
                    self.assertEqual(len(nodes), 1)
                    self.assertEqual(nodes[0].args[8], kind | 4)
                    self.assertFalse(
                        any(
                            "activation_cluster" in str(n.target)
                            for n in ep.graph.nodes
                        )
                    )
                    self.assertTrue(torch.all(ep.module()(data) >= 0))
        for activation in (torch.nn.ReLU(), torch.nn.ReLU6()):
            data = torch.randn(2, 3, 7)
            ep = self.lower(
                torch.nn.Sequential(torch.nn.Linear(7, 5), activation), data
            ).exported_program()
            node = next(
                n for n in ep.graph.nodes if "pulp_dynamic.quantized_" in str(n.target)
            )
            self.assertEqual(node.args[8], 5)
            self.assertTrue(torch.all(ep.module()(data) >= 0))

    def test_signed_fused_reference_bounds(self):
        from executorch.backends.pulp_dynamic.ops.operators import reference

        data = torch.tensor([[-128, -1, 0, 1, 127]], dtype=torch.int8)
        weight = torch.eye(5, dtype=torch.int8)
        config = torch.tensor(PulpDynamicConfig().serialized(), dtype=torch.int64)
        args = (data, weight, None, [1, 1], [0, 0], 0, 0)
        torch.testing.assert_close(reference(*args, config, 1, True), data)
        torch.testing.assert_close(reference(*args, config, 5, True), data.clamp_min(0))
        for maximum in (0, 6, 127):
            bounded = torch.cat((config, torch.tensor([maximum + 1])))
            torch.testing.assert_close(
                reference(*args, bounded, 5, True), data.clamp(0, maximum)
            )

    def test_unfused_activation_patterns(self):
        class Shared(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 1)

            def forward(self, x):
                y = self.conv(x)
                return y + torch.relu(y)

        data = torch.randn(1, 3, 5, 5).contiguous(memory_format=torch.channels_last)
        for model in (
            Shared(),
            torch.nn.Sequential(torch.nn.Conv2d(3, 5, 1), torch.nn.Hardtanh(-1, 1)),
        ):
            ep = self.lower(
                model.to(memory_format=torch.channels_last), data
            ).exported_program()
            node = next(
                n for n in ep.graph.nodes if "pulp_dynamic.quantized_" in str(n.target)
            )
            self.assertEqual(node.args[8], 1)
            self.assertTrue(
                any("activation_cluster" in str(n.target) for n in ep.graph.nodes)
            )

    def test_unsupported_pointwise_stride_and_small_budget_remain_portable(self):
        for model, config in (
            (torch.nn.Conv2d(8, 8, 1, stride=2), None),
            (torch.nn.Conv2d(128, 8, 3), PulpDynamicConfig(l1_bytes=256)),
        ):
            data = torch.randn(1, model.in_channels, 8, 8).contiguous(
                memory_format=torch.channels_last
            )
            edge = self.lower(
                model.to(memory_format=torch.channels_last), data, config=config
            )
            self.assertFalse(
                any(
                    "pulp_dynamic.quantized" in str(n.target)
                    for n in edge.exported_program().graph.nodes
                )
            )
            torch.testing.assert_close(
                edge.exported_program().module()(data), model(data)
            )
            self.assertTrue(edge.to_executorch().buffer)

    def test_portable_permute_layout_boundary_and_dynamic_view(self):
        class Rows(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 4)

            def forward(self, x):
                return self.linear(x.permute(0, 2, 3, 1).contiguous().reshape(-1, 3))

        data = torch.randn(1, 3, 3, 3).contiguous(memory_format=torch.channels_last)
        spec = (
            {
                2: torch.export.Dim("rows_h", min=2, max=4),
                3: torch.export.Dim("rows_w", min=2, max=4),
            },
        )
        edge = self.lower(Rows(), data, spec)
        targets = [str(n.target) for n in edge.exported_program().graph.nodes]
        self.assertTrue(any("dim_order_ops._to_dim_order_copy" in t for t in targets))
        self.assertEqual(edge.exported_program().module()(data).shape, (9, 4))
        self.assertTrue(edge.to_executorch().buffer)

    def test_eval_batchnorm_fold_uses_graph_connectivity(self):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4), torch.nn.ReLU()
        ).eval()
        data = torch.randn(1, 3, 8, 8).contiguous(memory_format=torch.channels_last)
        prepared = prepare_for_pulp_dynamic_export(model)
        torch.testing.assert_close(prepared(data), model(data))
        self.assertFalse(
            any(
                n.op == "call_module"
                and isinstance(prepared.get_submodule(n.target), torch.nn.BatchNorm2d)
                for n in prepared.graph.nodes
            )
        )


if __name__ == "__main__":
    unittest.main()
