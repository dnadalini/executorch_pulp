# Copyright (c) 2026 KU Leuven.
"""Dynamic spatial-mean lowering and the real channel-tiled C implementation."""

import ctypes
import shutil
import subprocess
import unittest
from pathlib import Path

import torch
from executorch.backends.pulp.edge_compile_config import pulp_edge_compile_config
from executorch.backends.pulp.passes.pulp_pass_manager import PulpPassManager
from executorch.backends.pulp.test import test_quantization_cluster as host_fixture
from executorch.exir import to_edge


class TestGlobalPoolLowering(unittest.TestCase):
    def test_dynamic_layouts_keepdim_and_serialization(self):
        class Pool(torch.nn.Module):
            def __init__(self, keepdim):
                super().__init__()
                self.keepdim = keepdim

            def forward(self, x):
                return x.mean(dim=(-1, -2), keepdim=self.keepdim)

        torch.set_num_threads(1)
        for layout in (torch.contiguous_format, torch.channels_last):
            for keepdim in (False, True):
                model = Pool(keepdim)
                data = torch.randn(2, 13, 7, 9).contiguous(memory_format=layout)
                ep = torch.export.export(
                    model,
                    (data,),
                    dynamic_shapes=(
                        {
                            2: torch.export.Dim("height", min=1, max=11),
                            3: torch.export.Dim("width", min=1, max=13),
                        },
                    ),
                )
                edge = to_edge(ep, compile_config=pulp_edge_compile_config()).transform(
                    PulpPassManager()
                )
                graph = edge.exported_program().graph
                self.assertEqual(
                    sum("pulp.global_avg_pool2d" in str(n.target) for n in graph.nodes),
                    1,
                )
                self.assertFalse(any("aten.mean" in str(n.target) for n in graph.nodes))
                for h, w in ((1, 1), (3, 5), (11, 13)):
                    sample = torch.randn(2, 13, h, w).contiguous(memory_format=layout)
                    torch.testing.assert_close(
                        edge.exported_program().module()(sample), model(sample)
                    )
                program = edge.to_executorch()
                self.assertTrue(program.buffer)
                operators = [
                    op.name
                    for op in program.executorch_program.execution_plan[0].operators
                ]
                self.assertIn("pulp::global_avg_pool2d", operators)
                self.assertNotIn("aten::mean", operators)
                self.assertFalse(any("dim_order_copy" in op for op in operators))

    def test_adaptive_pool_and_other_means(self):
        for model, expected in (
            (torch.nn.AdaptiveAvgPool2d(1), True),
            (torch.nn.AdaptiveAvgPool2d((2, 2)), False),
        ):
            data = torch.rand(1, 3, 6, 8).contiguous(memory_format=torch.channels_last)
            edge = to_edge(
                torch.export.export(model, (data,)),
                compile_config=pulp_edge_compile_config(),
            ).transform(PulpPassManager())
            self.assertEqual(
                any(
                    "pulp.global_avg_pool2d" in str(n.target)
                    for n in edge.exported_program().graph.nodes
                ),
                expected,
            )
            torch.testing.assert_close(
                edge.exported_program().module()(data), model(data)
            )

        class OtherMean(torch.nn.Module):
            def forward(self, x):
                return x.mean(dim=1), x.mean(dim=(2, 3), dtype=torch.float64)

        data = torch.rand(1, 3, 4, 5)
        edge = to_edge(
            torch.export.export(OtherMean(), (data,)),
            compile_config=pulp_edge_compile_config(),
        ).transform(PulpPassManager())
        self.assertFalse(
            any(
                "pulp.global_avg_pool2d" in str(n.target)
                for n in edge.exported_program().graph.nodes
            )
        )
        torch.testing.assert_close(
            edge.exported_program().module()(data), OtherMean()(data)
        )

    def test_out_resize_and_invalid_inputs(self):
        for keepdim in (False, True):
            out = torch.empty((1, 1, 1, 1) if keepdim else (1, 1))
            data = torch.randn(2, 7, 3, 9).contiguous(memory_format=torch.channels_last)
            torch.ops.pulp.global_avg_pool2d.out(data, keepdim, out=out)
            torch.testing.assert_close(out, data.mean((2, 3), keepdim=keepdim))
        for data in (
            torch.rand(1, 3, 2),
            torch.rand(1, 3, 0, 4),
            torch.ones(1, 3, 2, 4, dtype=torch.uint8),
        ):
            with self.assertRaises(ValueError):
                torch.ops.pulp.global_avg_pool2d(data)


class PoolArgs(ctypes.Structure):
    _fields_ = [
        ("input", ctypes.c_void_p),
        ("output", ctypes.c_void_p),
        ("batches", ctypes.c_size_t),
        ("channels", ctypes.c_size_t),
        ("spatial", ctypes.c_size_t),
        ("channels_last", ctypes.c_int),
        ("status", ctypes.c_int),
    ]


@unittest.skipUnless(shutil.which("cc"), "Host C compiler is required")
class TestGlobalPoolWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        host_fixture.TestClusterQuantization.setUpClass.__func__(cls)
        directory = Path(cls.directory.name)
        with (directory / "pmsis.h").open("a") as file:
            file.write(
                r"""
static inline void pi_cl_dma_cmd_2d(uintptr_t ext, uintptr_t loc, size_t size,
    size_t stride, size_t length, int direction, pi_cl_dma_cmd_t *cmd) {
  (void)cmd; ++host_dmas;
  while(size) {
    size_t chunk=size<length?size:length;
    if(direction) memcpy((void*)loc,(void*)ext,chunk);
    else memcpy((void*)ext,(void*)loc,chunk);
    ext+=stride;loc+=chunk;size-=chunk;
  }
}
"""
            )
        ops = Path(__file__).resolve().parents[1] / "ops"
        output = directory / "pool.so"
        subprocess.run(
            [
                "cc",
                "-shared",
                "-fPIC",
                "-O2",
                "-ffp-contract=off",
                "-fno-fast-math",
                "-DNUM_CORES=8",
                "-DPULP_GLOBAL_POOL_L1_BYTES=256",
                "-I" + str(directory),
                "-I" + str(ops),
                str(ops / "pulp_global_pool_cluster.c"),
                str(directory / "shim.c"),
                "-o",
                str(output),
            ],
            check=True,
        )
        cls.lib = ctypes.CDLL(str(output))
        cls.lib.pulp_cluster_global_pool_entry.argtypes = [ctypes.POINTER(PoolArgs)]

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        host_fixture.TestClusterQuantization.setUp(self)

    def execute(self, source, limit=None):
        n, c, h, w = source.shape
        storage = torch.full((n * c + 8,), 123.0)
        output = storage[4:-4].reshape(n, c)
        if limit is not None:
            ctypes.c_size_t.in_dll(self.lib, "host_limit").value = limit
        channels_last = not source.is_contiguous()
        args = PoolArgs(
            source.data_ptr(), output.data_ptr(), n, c, h * w, channels_last, -1
        )
        self.lib.pulp_cluster_global_pool_entry(ctypes.byref(args))
        self.assertTrue(
            torch.all(storage[:4] == 123) and torch.all(storage[-4:] == 123)
        )
        self.assertLessEqual(ctypes.c_size_t.in_dll(self.lib, "host_peak").value, 256)
        return args.status, output

    def test_channels_spatial_chunks_layouts_and_batches(self):
        torch.set_num_threads(1)
        torch.manual_seed(43)
        for shape in (
            (1, 1, 1, 1),
            (1, 7, 3, 5),
            (2, 13, 7, 11),
            (1, 33, 2, 2),
            (1, 1280, 1, 1),
            (1, 9, 101, 103),
            (1, 17000, 1, 2),
        ):
            for layout in (torch.contiguous_format, torch.channels_last):
                with self.subTest(shape=shape, layout=layout):
                    data = torch.randn(shape).contiguous(memory_format=layout)
                    status, output = self.execute(data)
                    self.assertEqual(status, 0)
                    torch.testing.assert_close(
                        output, data.mean((2, 3)), atol=1e-6, rtol=1e-5
                    )
                    self.assertGreater(
                        ctypes.c_uint.in_dll(self.lib, "host_forks").value, 0
                    )

    def test_retries_empty_and_allocation_failure(self):
        data = torch.randn(1, 13, 5, 7)
        status, output = self.execute(data, limit=128)
        self.assertEqual(status, 0)
        torch.testing.assert_close(output, data.mean((2, 3)))
        status, _ = self.execute(data, limit=1)
        self.assertEqual(status, -1)
        status, _ = self.execute(torch.empty(0, 13, 5, 7))
        self.assertEqual(status, 0)
        status, _ = self.execute(torch.empty(1, 13, 0, 7))
        self.assertEqual(status, -1)


if __name__ == "__main__":
    unittest.main()
