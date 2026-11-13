# Copyright (c) 2026 KU Leuven.
"""Check DMA tiling and the actual C worker against the Python reference."""

import ctypes
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import executorch.backends.pulp.ops.operators  # noqa: F401

import torch


class Args(ctypes.Structure):
    _fields_ = [
        ("input", ctypes.c_void_p),
        ("output", ctypes.c_void_p),
        ("count", ctypes.c_size_t),
        ("scale", ctypes.c_float),
        ("inverse_scale", ctypes.c_float),
        ("zero_point", ctypes.c_int32),
        ("quant_min", ctypes.c_int32),
        ("quant_max", ctypes.c_int32),
        ("quantize", ctypes.c_int),
        ("status", ctypes.c_int),
    ]


@unittest.skipUnless(shutil.which("cc"), "Host C compiler is required")
class TestClusterQuantization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        directory = Path(cls.directory.name)
        ops = Path(__file__).resolve().parents[1] / "ops"
        # uintptr_t keeps full host addresses; on the device these DMA arguments
        # are 32-bit. Every DMA is tracked and copied using the requested length.
        (directory / "pmsis.h").write_text(
            r"""
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
extern unsigned host_core, host_dmas, host_forks;
extern size_t host_limit, host_peak;
typedef struct { int unused; } pi_cl_dma_cmd_t;
#define PI_CL_DMA_DIR_EXT2LOC 1
#define PI_CL_DMA_DIR_LOC2EXT 0
static inline unsigned pi_core_id(void) { return host_core; }
static inline void *pi_cl_l1_malloc(int n, size_t size) {
  (void)n; if (size > host_limit) return NULL;
  if (size > host_peak) host_peak = size; return malloc(size);
}
static inline void pi_cl_l1_free(int n, void *p, size_t size) { (void)n; (void)size; free(p); }
static inline void pi_cl_team_fork(unsigned n, void (*worker)(void*), void *p) {
  ++host_forks; for (host_core=0; host_core<n; ++host_core) worker(p);
}
static inline void pi_cl_dma_cmd(uintptr_t ext, uintptr_t loc, size_t size, int direction, pi_cl_dma_cmd_t *cmd) {
  (void)cmd; ++host_dmas;
  if (direction) memcpy((void*)loc, (void*)ext, size);
  else memcpy((void*)ext, (void*)loc, size);
}
static inline void pi_cl_dma_cmd_wait(pi_cl_dma_cmd_t *cmd) { (void)cmd; }
"""
        )
        (directory / "shim.c").write_text(
            "#include <stddef.h>\n"
            "unsigned host_core, host_dmas, host_forks;\n"
            "size_t host_limit = (size_t)-1, host_peak;\n"
        )
        library = directory / "quantization.so"
        subprocess.run(
            [
                "cc",
                "-shared",
                "-fPIC",
                "-O2",
                "-ffp-contract=off",
                "-fno-fast-math",
                "-DNUM_CORES=8",
                "-DPULP_QUANTIZATION_L1_BYTES=256",
                "-I" + str(directory),
                "-I" + str(ops),
                str(ops / "pulp_quantization_cluster.c"),
                str(directory / "shim.c"),
                "-o",
                str(library),
            ],
            check=True,
        )
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.pulp_cluster_quantization_entry.argtypes = [ctypes.POINTER(Args)]

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        for name in ("host_dmas", "host_forks"):
            ctypes.c_uint.in_dll(self.lib, name).value = 0
        ctypes.c_size_t.in_dll(self.lib, "host_peak").value = 0
        ctypes.c_size_t.in_dll(self.lib, "host_limit").value = ctypes.c_size_t(-1).value

    def run_worker(self, source, scale, zp, qmin, qmax, quantize):
        source = source.contiguous()
        # Guard both ends of the output to detect partial-tile overstores.
        output = torch.full(
            (source.numel() + 8,), 123, dtype=torch.uint8 if quantize else torch.float32
        )
        view = output[4:-4]
        args = Args(
            source.data_ptr(),
            view.data_ptr(),
            source.numel(),
            scale,
            float(1.0 / scale),
            zp,
            qmin,
            qmax,
            quantize,
            -1,
        )
        self.lib.pulp_cluster_quantization_entry(ctypes.byref(args))
        self.assertTrue(torch.all(output[:4] == 123))
        self.assertTrue(torch.all(output[-4:] == 123))
        self.assertLessEqual(ctypes.c_size_t.in_dll(self.lib, "host_peak").value, 256)
        return args.status, view

    def test_rounding_clipping_zero_points_and_many_tiles(self):
        for count in (0, 1, 3, 7, 31, 32, 33, 257, 65537):
            for scale, zp, qmin, qmax in (
                (0.25, 0, 0, 255),
                (0.125, 128, 10, 200),
                (0.037, 17, 0, 255),
            ):
                with self.subTest(count=count, scale=scale, zp=zp):
                    raw = (
                        (torch.arange(count, dtype=torch.float32) % 600) - 150.5
                    ) * scale
                    status, result = self.run_worker(raw, scale, zp, qmin, qmax, True)
                    self.assertEqual(status, 0)
                    expected = torch.ops.pulp.quantize_per_tensor(
                        raw, scale, zp, qmin, qmax, torch.uint8
                    )
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)
                    status, restored = self.run_worker(
                        result, scale, zp, qmin, qmax, False
                    )
                    self.assertEqual(status, 0)
                    expected = torch.ops.pulp.dequantize_per_tensor(
                        result, scale, zp, qmin, qmax, torch.uint8
                    )
                    torch.testing.assert_close(restored, expected, rtol=0, atol=0)
        self.assertGreater(ctypes.c_uint.in_dll(self.lib, "host_dmas").value, 100)

    def test_infinity_nan_and_allocation_failure(self):
        status, result = self.run_worker(
            torch.tensor([-float("inf"), float("inf")]), 0.5, 0, 0, 255, True
        )
        self.assertEqual(status, 0)
        self.assertEqual(result.tolist(), [0, 255])
        status, _ = self.run_worker(torch.tensor([float("nan")]), 0.5, 0, 0, 255, True)
        self.assertEqual(status, -2)
        ctypes.c_size_t.in_dll(self.lib, "host_limit").value = 1
        status, _ = self.run_worker(torch.ones(123), 0.5, 0, 0, 255, True)
        self.assertEqual(status, -1)

    def test_retries_smaller_tiles_and_preserves_tails(self):
        ctypes.c_size_t.in_dll(self.lib, "host_limit").value = 120
        source = torch.linspace(-1, 30, 99)
        status, result = self.run_worker(source, 0.25, 0, 0, 255, True)
        self.assertEqual(status, 0)
        torch.testing.assert_close(
            result,
            torch.ops.pulp.quantize_per_tensor(source, 0.25, 0, 0, 255, torch.uint8),
        )
        self.assertLessEqual(ctypes.c_size_t.in_dll(self.lib, "host_peak").value, 120)
        self.assertEqual(
            ctypes.c_uint.in_dll(self.lib, "host_dmas").value,
            2 * ctypes.c_uint.in_dll(self.lib, "host_forks").value,
        )

    def test_invalid_parameters_match_native_contract(self):
        for scale in (0.0, -1.0, float("inf"), float("nan"), 1e40, 1e-50):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                torch.ops.pulp.quantize_per_tensor_cluster(
                    torch.ones(3), scale, 0, 0, 255, torch.uint8
                )
        for zp, qmin, qmax in ((0, -1, 255), (0, 0, 256), (3, 4, 100), (101, 4, 100)):
            with (
                self.subTest(zp=zp, qmin=qmin, qmax=qmax),
                self.assertRaises(ValueError),
            ):
                torch.ops.pulp.quantize_per_tensor_cluster(
                    torch.ones(3), 0.25, zp, qmin, qmax, torch.uint8
                )
        with self.assertRaises(ValueError):
            torch.ops.pulp.quantize_per_tensor_cluster(
                torch.ones(3, dtype=torch.float16), 0.25, 0, 0, 255, torch.uint8
            )
        with self.assertRaises(ValueError):
            torch.ops.pulp.dequantize_per_tensor_cluster(
                torch.ones(3, dtype=torch.int8), 0.25, 0, 0, 255, torch.uint8
            )

    def test_named_operators_reference_and_resize(self):
        out = torch.empty(0, dtype=torch.uint8)
        restored = torch.empty(0)
        for count in (257, 3, 513):
            data = torch.arange(count).float() / 8
            expected = torch.ops.pulp.quantize_per_tensor(
                data, 0.25, 0, 0, 255, torch.uint8
            )
            torch.ops.pulp.quantize_per_tensor_cluster.out(
                data, 0.25, 0, 0, 255, torch.uint8, out=out
            )
            torch.testing.assert_close(out, expected)
            torch.ops.pulp.dequantize_per_tensor_cluster.out(
                out, 0.25, 0, 0, 255, torch.uint8, out=restored
            )
            torch.testing.assert_close(restored, expected.float() * 0.25)
        with self.assertRaises(ValueError):
            torch.ops.pulp.quantize_per_tensor_cluster(
                torch.tensor([float("nan")]), 0.25, 0, 0, 255, torch.uint8
            )


if __name__ == "__main__":
    unittest.main()
