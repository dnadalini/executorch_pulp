# Copyright (c) 2026 KU Leuven.
"""Activation fusion, conversion boundaries, and the DMA-tiled C workers."""

import ctypes
import math
import shutil
import subprocess
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from executorch.backends.pulp.edge_compile_config import pulp_edge_compile_config
from executorch.backends.pulp.passes.pulp_pass_manager import PulpPassManager
from executorch.backends.pulp.quantizer.quantizer import PulpQuantizer
from executorch.backends.pulp.test import (
    test_quantization_cluster as quantization_fixture,
)
from executorch.exir import to_edge
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


def lower(model, data, quantize=True, dynamic_shapes=None):
    model = model.eval()
    if quantize:
        prepared = prepare_pt2e(
            torch.export.export(model, (data,)).module(), PulpQuantizer()
        )
        prepared(data)
        model = convert_pt2e(prepared)
    return to_edge(
        torch.export.export(model, (data,), dynamic_shapes=dynamic_shapes),
        compile_config=pulp_edge_compile_config(),
    ).transform(PulpPassManager())


class TestActivationLowering(unittest.TestCase):
    def test_affine_fusions_and_serialization(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)
        for mode in ("conv", "depthwise", "pointwise", "linear"):
            for activation in (
                torch.nn.ReLU(),
                torch.nn.ReLU6(),
                torch.nn.Hardtanh(0, 6),
            ):
                with self.subTest(mode=mode, activation=activation):
                    linear = mode == "linear"
                    affine = (
                        torch.nn.Linear(5, 7)
                        if linear
                        else torch.nn.Conv2d(
                            5,
                            5 if mode == "depthwise" else 7,
                            1 if mode == "pointwise" else 3,
                            padding=0 if mode == "pointwise" else 1,
                            groups=5 if mode == "depthwise" else 1,
                        )
                    )
                    # Exercise an effective ReLU6 cap, not just identity clipping.
                    with torch.no_grad():
                        affine.bias.fill_(20)
                    data = (
                        torch.rand(2, 5) * 10
                        if linear
                        else (torch.rand(1, 5, 4, 7) * 10).contiguous(
                            memory_format=torch.channels_last
                        )
                    )
                    edge = lower(torch.nn.Sequential(affine, activation), data)
                    ep = edge.exported_program()
                    nodes = [
                        n for n in ep.graph.nodes if "pulp.quantized_" in str(n.target)
                    ]
                    self.assertEqual(len(nodes), 1)
                    self.assertFalse(
                        any(
                            "aten.relu" in str(n.target)
                            or "aten.hardtanh" in str(n.target)
                            or "activation_cluster" in str(n.target)
                            for n in ep.graph.nodes
                        )
                    )
                    node = nodes[0]
                    if isinstance(activation, torch.nn.ReLU):
                        self.assertNotIn("clamp", str(node.target))
                    else:
                        self.assertIn("clamp", str(node.target))
                        self.assertLess(node.args[-1], 255)
                        buffers = ep.graph_signature.inputs_to_buffers
                        scale = ep.state_dict[buffers[node.args[3].name]][2].item()
                        expected_max = round(float(torch.tensor(6.0) * (1.0 / scale)))
                        self.assertEqual(node.args[-1], expected_max)
                        self.assertLessEqual(
                            ep.module()(data).max().item(), expected_max * scale + 1e-5
                        )
                    self.assertTrue(edge.to_executorch().buffer)

    def test_branched_convolution_is_not_clamped(self):
        class Branched(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(4, 4, 1)

            def forward(self, x):
                y = self.conv(x)
                return y, F.hardtanh(y, 0, 6)

        data = torch.rand(1, 4, 3, 5).contiguous(memory_format=torch.channels_last)
        ep = lower(Branched(), data).exported_program()
        self.assertFalse(any("_clamp" in str(n.target) for n in ep.graph.nodes))
        self.assertEqual(
            sum("activation_cluster" in str(n.target) for n in ep.graph.nodes), 1
        )
        raw, activated = ep.module()(data)
        torch.testing.assert_close(activated, raw.clamp(0, 6))

    def test_float_activations_dynamic_shapes_and_gelu_modes(self):
        for activation in (
            torch.nn.ReLU(),
            torch.nn.ReLU6(),
            torch.nn.Hardtanh(-2, 3),
            torch.nn.GELU(),
            torch.nn.GELU(approximate="tanh"),
        ):
            data = torch.randn(3, 9)
            edge = lower(
                activation,
                data,
                quantize=False,
                dynamic_shapes=({0: torch.export.Dim("rows", min=1, max=5)},),
            )
            ep = edge.exported_program()
            self.assertEqual(
                sum("activation_cluster" in str(n.target) for n in ep.graph.nodes), 1
            )
            for rows in (1, 3, 5):
                sample = torch.randn(rows, 9)
                torch.testing.assert_close(ep.module()(sample), activation(sample))
            self.assertTrue(edge.to_executorch().buffer)

    def test_standalone_absorbs_quantization_boundaries(self):
        class QuantizedActivation(torch.nn.Module):
            def forward(self, x):
                y = torch.ops.pulp.dequantize_per_tensor_cluster(
                    x, 0.125, 17, 0, 255, torch.uint8
                )
                y = F.gelu(y, approximate="tanh")
                return torch.ops.pulp.quantize_per_tensor_cluster(
                    y, 0.037, 9, 0, 255, torch.uint8
                )

        model = QuantizedActivation()
        data = torch.arange(255, dtype=torch.uint8)
        edge = lower(model, data, quantize=False)
        calls = [
            n for n in edge.exported_program().graph.nodes if n.op == "call_function"
        ]
        self.assertEqual(len(calls), 1)
        self.assertIn("activation_cluster", str(calls[0].target))
        torch.testing.assert_close(edge.exported_program().module()(data), model(data))
        self.assertTrue(edge.to_executorch().buffer)


class ActivationArgs(ctypes.Structure):
    _fields_ = [
        ("input", ctypes.c_void_p),
        ("output", ctypes.c_void_p),
        ("count", ctypes.c_size_t),
        *[
            (name, ctypes.c_float)
            for name in ("lower", "upper", "input_scale", "output_inverse_scale")
        ],
        *[
            (name, ctypes.c_int32)
            for name in (
                "input_zero_point",
                "output_zero_point",
                "quant_min",
                "quant_max",
            )
        ],
        *[
            (name, ctypes.c_int)
            for name in ("kind", "input_quantized", "output_quantized", "status")
        ],
    ]


@unittest.skipUnless(shutil.which("cc"), "Host C compiler is required")
class TestActivationWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the PMSIS DMA/multicore host fixture, compile the actual worker.
        quantization_fixture.TestClusterQuantization.setUpClass.__func__(cls)
        directory = Path(cls.directory.name)
        ops = Path(__file__).resolve().parents[1] / "ops"
        output = directory / "activation.so"
        subprocess.run(
            [
                "cc",
                "-shared",
                "-fPIC",
                "-O2",
                "-ffp-contract=off",
                "-fno-fast-math",
                "-DNUM_CORES=8",
                "-DPULP_ACTIVATION_L1_BYTES=256",
                "-I" + str(directory),
                "-I" + str(ops),
                str(ops / "pulp_activation_cluster.c"),
                str(directory / "shim.c"),
                "-lm",
                "-o",
                str(output),
            ],
            check=True,
        )
        cls.lib = ctypes.CDLL(str(output))
        cls.lib.pulp_cluster_activation_entry.argtypes = [
            ctypes.POINTER(ActivationArgs)
        ]

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        quantization_fixture.TestClusterQuantization.setUp(self)

    def run_worker(self, source, kind, output_quantized, limit=None):
        source = source.contiguous()
        result = torch.full(
            (source.numel() + 8,),
            123,
            dtype=torch.uint8 if output_quantized else torch.float32,
        )
        output = result[4:-4]
        if limit is not None:
            ctypes.c_size_t.in_dll(self.lib, "host_limit").value = limit
        args = ActivationArgs(
            source.data_ptr(),
            output.data_ptr(),
            source.numel(),
            0 if kind == 0 else -2,
            math.inf if kind == 0 else 6,
            0.125,
            1.0 / 0.037,
            17,
            9,
            0,
            255,
            kind,
            source.dtype == torch.uint8,
            output_quantized,
            -1,
        )
        self.lib.pulp_cluster_activation_entry(ctypes.byref(args))
        self.assertTrue(torch.all(result[:4] == 123) and torch.all(result[-4:] == 123))
        self.assertLessEqual(ctypes.c_size_t.in_dll(self.lib, "host_peak").value, 256)
        return args.status, output

    def test_tiles_tails_conversions_and_gelu(self):
        for count in (0, 1, 3, 7, 33, 257, 65537):
            for kind in range(4):
                for input_quantized in (False, True):
                    for output_quantized in (False, True):
                        with self.subTest(
                            count=count,
                            kind=kind,
                            input_quantized=input_quantized,
                            output_quantized=output_quantized,
                        ):
                            source = (
                                torch.arange(count, dtype=torch.int32)
                                .remainder(255)
                                .to(torch.uint8)
                                if input_quantized
                                else torch.linspace(-10, 10, count)
                            )
                            status, actual = self.run_worker(
                                source, kind, output_quantized
                            )
                            expected = torch.ops.pulp.activation_cluster(
                                source,
                                kind,
                                0.0 if kind == 0 else -2.0,
                                math.inf if kind == 0 else 6.0,
                                0.125,
                                17,
                                0.037,
                                9,
                                0,
                                255,
                                output_quantized,
                            )
                            self.assertEqual(status, 0)
                            if output_quantized:
                                self.assertLessEqual(
                                    (
                                        (actual.int() - expected.int())
                                        .abs()
                                        .max()
                                        .item()
                                        if count
                                        else 0
                                    ),
                                    0 if kind < 2 else 1,
                                )
                            else:
                                torch.testing.assert_close(
                                    actual, expected, rtol=1e-5, atol=2e-6
                                )
                            if count:
                                self.assertEqual(
                                    ctypes.c_uint.in_dll(self.lib, "host_dmas").value,
                                    2
                                    * ctypes.c_uint.in_dll(
                                        self.lib, "host_forks"
                                    ).value,
                                )

    def test_allocator_retry_failure_and_nan(self):
        source = torch.linspace(-8, 8, 257)
        status, result = self.run_worker(source, 1, False, limit=224)
        self.assertEqual(status, 0)
        torch.testing.assert_close(result, source.clamp(-2, 6))
        status, _ = self.run_worker(source, 1, False, limit=1)
        self.assertEqual(status, -1)
        ctypes.c_size_t.in_dll(self.lib, "host_limit").value = ctypes.c_size_t(-1).value
        status, _ = self.run_worker(torch.tensor([math.nan]), 0, True)
        self.assertEqual(status, -2)
        status, result = self.run_worker(
            torch.tensor([math.nan, -math.inf, math.inf]), 1, False
        )
        self.assertEqual(status, 0)
        torch.testing.assert_close(
            result, torch.tensor([math.nan, -2, 6]), equal_nan=True
        )


if __name__ == "__main__":
    unittest.main()
