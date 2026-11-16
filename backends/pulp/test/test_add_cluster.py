# Copyright (c) 2026 KU Leuven.
"""Elementwise-add lowering, residual layout copies, and real tiled C workers."""

import ctypes
import math
import shutil
import subprocess
import unittest
from pathlib import Path

import torch
from executorch.backends.pulp.edge_compile_config import pulp_edge_compile_config
from executorch.backends.pulp.export import pulp_export_context
from executorch.backends.pulp.passes.pulp_pass_manager import PulpPassManager
from executorch.backends.pulp.test import test_quantization_cluster as fixture
from executorch.exir import to_edge


class Residual(torch.nn.Module):
    def forward(self, x, y):
        a = torch.ops.aten._to_copy.default(x, memory_format=torch.contiguous_format)
        b = torch.ops.aten._to_copy.default(y, memory_format=torch.contiguous_format)
        return torch.ops.aten._to_copy.default(a + b, memory_format=torch.channels_last)


class TestAddLowering(unittest.TestCase):
    def test_residual_copies_removed_for_dynamic_and_singleton_shapes(self):
        for shape, dynamic in (((1, 7, 5, 7), True), ((1, 7, 1, 1), False)):
            x = torch.randn(shape).to(memory_format=torch.channels_last)
            y = torch.randn_like(x)
            with pulp_export_context():
                spatial = {
                    2: torch.export.Dim("h", min=1, max=9),
                    3: torch.export.Dim("w", min=1, max=9),
                }
                exported = torch.export.export(
                    Residual(),
                    (x, y),
                    dynamic_shapes=(spatial, spatial) if dynamic else None,
                )
                edge = to_edge(
                    exported, compile_config=pulp_edge_compile_config()
                ).transform(PulpPassManager())
                calls = [
                    n
                    for n in edge.exported_program().graph.nodes
                    if n.op == "call_function"
                ]
                self.assertEqual(len(calls), 1)
                self.assertIn("pulp.add_cluster", str(calls[0].target))
                for h, w in ((1, 1), (3, 7), (9, 9)) if dynamic else ((1, 1),):
                    a = torch.randn(1, 7, h, w).to(memory_format=torch.channels_last)
                    b = torch.randn_like(a)
                    actual = edge.exported_program().module()(a, b)
                    torch.testing.assert_close(actual, a + b, rtol=0, atol=0)
                    self.assertEqual(actual.dim_order(), (0, 2, 3, 1))
                program = edge.to_executorch()
                self.assertEqual(
                    [
                        (op.name, op.overload)
                        for op in program.executorch_program.execution_plan[0].operators
                    ],
                    [("pulp::add_cluster", "out")],
                )

    def test_scaled_add_default_layout_and_broadcast_fallback(self):
        class Add(torch.nn.Module):
            def forward(self, x, y):
                return torch.add(x, y, alpha=-0.5)

        for shape, other_shape, expected in (
            ((2, 3, 5), (2, 3, 5), True),
            ((2, 3, 5), (1, 3, 1), False),
        ):
            x, y = torch.randn(shape), torch.randn(other_shape)
            edge = to_edge(
                torch.export.export(Add(), (x, y)),
                compile_config=pulp_edge_compile_config(),
            ).transform(PulpPassManager())
            self.assertEqual(
                any(
                    "pulp.add_cluster" in str(n.target)
                    for n in edge.exported_program().graph.nodes
                ),
                expected,
            )
            torch.testing.assert_close(
                edge.exported_program().module()(x, y), Add()(x, y)
            )
            self.assertTrue(edge.to_executorch().buffer)

    def test_shared_copy_consumer_keeps_its_requested_layout(self):
        class Shared(torch.nn.Module):
            def forward(self, x, y):
                copy = torch.ops.aten._to_copy.default(
                    x, memory_format=torch.contiguous_format
                )
                other = torch.ops.aten._to_copy.default(
                    y, memory_format=torch.contiguous_format
                )
                added = torch.ops.aten._to_copy.default(
                    copy + other, memory_format=torch.channels_last
                )
                return copy, added

        x = torch.randn(1, 5, 3, 7).to(memory_format=torch.channels_last)
        y = torch.randn_like(x)
        edge = to_edge(
            torch.export.export(Shared(), (x, y)),
            compile_config=pulp_edge_compile_config(),
        ).transform(PulpPassManager())
        a, b = edge.exported_program().module()(x, y)
        torch.testing.assert_close(a, x)
        torch.testing.assert_close(b, x + y)
        self.assertEqual(a.dim_order(), (0, 1, 2, 3))
        self.assertEqual(b.dim_order(), (0, 2, 3, 1))
        self.assertEqual(
            sum(
                "_to_dim_order_copy" in str(n.target)
                for n in edge.exported_program().graph.nodes
            ),
            1,
        )

    def test_reference_output_resize_and_type_checks(self):
        for shape, fmt in (
            ((2, 3, 7), torch.contiguous_format),
            ((1, 3, 1, 1), torch.channels_last),
            ((1, 3, 5, 7), torch.channels_last),
        ):
            x = torch.randn(shape).contiguous(memory_format=fmt)
            y = torch.randn_like(x)
            out = torch.empty((1,) * len(shape), memory_format=fmt)
            torch.ops.pulp.add_cluster.out(x, y, 0.125, out=out)
            torch.testing.assert_close(out, torch.add(x, y, alpha=0.125))
            self.assertEqual(out.dim_order(), x.dim_order())
        for a, b in (
            (torch.ones(2, dtype=torch.uint8), torch.ones(2, dtype=torch.uint8)),
            (torch.ones(2, 3), torch.ones(3)),
        ):
            with self.assertRaises(ValueError):
                torch.ops.pulp.add_cluster(a, b)


class Args(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_void_p),
        ("right", ctypes.c_void_p),
        ("output", ctypes.c_void_p),
        ("count", ctypes.c_size_t),
        ("alpha", ctypes.c_float),
        ("status", ctypes.c_int),
    ]


@unittest.skipUnless(shutil.which("cc"), "Host C compiler is required")
class TestAddWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.TestClusterQuantization.setUpClass.__func__(cls)
        folder = Path(cls.directory.name)
        # The shared DMA header also declares a gather helper, unused by add.
        with (folder / "pmsis.h").open("a") as file:
            file.write(
                "\nstatic inline void pi_cl_dma_cmd_2d(uintptr_t e,uintptr_t l,size_t s,size_t st,size_t len,int d,pi_cl_dma_cmd_t*c){(void)e;(void)l;(void)s;(void)st;(void)len;(void)d;(void)c;abort();}\n"
            )
        ops = Path(__file__).resolve().parents[1] / "ops"
        output = folder / "add.so"
        subprocess.run(
            [
                "cc",
                "-shared",
                "-fPIC",
                "-O2",
                "-ffp-contract=off",
                "-fno-fast-math",
                "-DNUM_CORES=8",
                "-DPULP_ADD_L1_BYTES=256",
                "-I" + str(folder),
                "-I" + str(ops),
                str(ops / "pulp_add_cluster.c"),
                str(folder / "shim.c"),
                "-o",
                str(output),
            ],
            check=True,
        )
        cls.lib = ctypes.CDLL(str(output))
        cls.lib.pulp_cluster_add_entry.argtypes = [ctypes.POINTER(Args)]

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        fixture.TestClusterQuantization.setUp(self)

    def execute(self, a, b, alpha=1.0, limit=None):
        a, b = a.contiguous(), b.contiguous()
        saved_a, saved_b = a.clone(), b.clone()
        storage = torch.full((a.numel() + 8,), 123.0)
        output = storage[4:-4]
        if limit is not None:
            ctypes.c_size_t.in_dll(self.lib, "host_limit").value = limit
        args = Args(a.data_ptr(), b.data_ptr(), output.data_ptr(), a.numel(), alpha, -1)
        self.lib.pulp_cluster_add_entry(ctypes.byref(args))
        self.assertTrue(
            torch.all(storage[:4] == 123) and torch.all(storage[-4:] == 123)
        )
        torch.testing.assert_close(a, saved_a, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(b, saved_b, rtol=0, atol=0, equal_nan=True)
        self.assertLessEqual(ctypes.c_size_t.in_dll(self.lib, "host_peak").value, 256)
        self.assertEqual(
            ctypes.c_uint.in_dll(self.lib, "host_dmas").value,
            3 * ctypes.c_uint.in_dll(self.lib, "host_forks").value,
        )
        return args.status, output

    def test_tiles_tails_scaled_values_and_nonfinite(self):
        torch.set_num_threads(1)
        for count in (0, 1, 3, 7, 31, 257, 65537):
            a, b = torch.linspace(-8, 8, count), torch.linspace(3, -5, count)
            for alpha in (1.0, 0.0, -2.0, 0.125):
                with self.subTest(count=count, alpha=alpha):
                    status, output = self.execute(a, b, alpha)
                    self.assertEqual(status, 0)
                    torch.testing.assert_close(
                        output,
                        torch.add(a, b, alpha=alpha),
                        rtol=0 if alpha == 1 else 1e-5,
                        atol=0 if alpha == 1 else 1e-6,
                    )
        a = torch.tensor([math.nan, math.inf, -math.inf, 1.0, -0.0])
        b = torch.tensor([1.0, -math.inf, math.inf, math.inf, -0.0])
        for alpha in (1.0, 0.0, -0.5):
            status, output = self.execute(a, b, alpha)
            self.assertEqual(status, 0)
            torch.testing.assert_close(
                output, torch.add(a, b, alpha=alpha), equal_nan=True
            )

    def test_allocation_retry_and_failure(self):
        a, b = torch.randn(257), torch.randn(257)
        status, output = self.execute(a, b, limit=160)
        self.assertEqual(status, 0)
        torch.testing.assert_close(output, a + b, rtol=0, atol=0)
        status, output = self.execute(a, b, limit=1)
        self.assertEqual(status, -1)
        self.assertTrue(torch.all(output == 123))


if __name__ == "__main__":
    unittest.main()
