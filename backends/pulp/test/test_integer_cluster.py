# Copyright (c) 2026 KU Leuven.
"""Exercise the actual integer C workers, DMA tiling and affine reference."""

import ctypes
import shutil
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import torch
from executorch.backends.pulp.ops.integer_ops import fixed_point
from executorch.backends.pulp.test import test_quantization_cluster as quantization_test


class Args(ctypes.Structure):
    _fields_ = (
        [(n, ctypes.c_void_p) for n in ("input", "other", "output")]
        + [(n, ctypes.c_size_t) for n in ("count", "batches", "channels", "spatial")]
        + [(n, ctypes.c_int64) for n in ("ma", "mb")]
        + [
            (n, ctypes.c_int)
            for n in (
                "shift",
                "za",
                "zb",
                "zo",
                "qmin",
                "qmax",
                "operation",
                "channels_last",
                "status",
            )
        ]
    )


@unittest.skipUnless(shutil.which("cc"), "Host C compiler required")
class TestIntegerCluster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the SDK-independent allocation/core substitutes, and use the
        # common DMA stub to enforce hardware transfer-size restrictions.
        quantization_test.TestClusterQuantization.setUpClass()
        cls.directory = tempfile.TemporaryDirectory()
        directory = Path(cls.directory.name)
        ops = Path(__file__).resolve().parents[1] / "ops"
        original = Path(quantization_test.TestClusterQuantization.directory.name)
        stub = (original / "pmsis.h").read_text()
        begin = stub.index("typedef struct")
        end = stub.index("static inline unsigned pi_core_id")
        stub = stub[:begin] + '#include "dma_host_stub.h"\n' + stub[end:]
        stub = stub[: stub.index("static inline void pi_cl_dma_cmd(")]
        (directory / "pmsis.h").write_text(stub)
        shutil.copy(Path(__file__).with_name("dma_host_stub.h"), directory)
        shutil.copy(original / "shim.c", directory)
        library = directory / "integer.so"
        subprocess.run(
            [
                "cc",
                "-shared",
                "-fPIC",
                "-O2",
                "-fsanitize=undefined",
                "-fno-sanitize-recover=undefined",
                "-DNUM_CORES=8",
                "-DPULP_INTEGER_L1_BYTES=512",
                "-I" + str(directory),
                "-I" + str(ops),
                str(ops / "pulp_integer_cluster.c"),
                str(directory / "shim.c"),
                "-o",
                str(library),
            ],
            check=True,
        )
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.pulp_cluster_integer_entry.argtypes = [ctypes.POINTER(Args)]
        quantization_test.TestClusterQuantization.tearDownClass()

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        ctypes.c_size_t.in_dll(self.lib, "host_limit").value = ctypes.c_size_t(-1).value
        ctypes.c_size_t.in_dll(self.lib, "host_peak").value = 0

    def run_worker(self, source, params, operation, other=None, limit=None):
        if limit is not None:
            ctypes.c_size_t.in_dll(self.lib, "host_limit").value = limit
        output_count = (
            source.shape[0] * source.shape[1] if operation == 2 else source.numel()
        )
        storage = torch.full((output_count + 8,), 123, dtype=torch.uint8)
        output = storage[4:-4]
        args = Args()
        args.input, args.output, args.count = (
            source.data_ptr(),
            output.data_ptr(),
            source.numel(),
        )
        args.other = other.data_ptr() if other is not None else None
        if operation == 2:
            args.batches, args.channels = source.shape[:2]
            args.spatial = source.shape[2] * source.shape[3]
            args.channels_last = source.dim_order() == (0, 2, 3, 1)
        if operation == 1:
            (
                args.ma,
                args.mb,
                args.shift,
                args.za,
                args.zb,
                args.zo,
                args.qmin,
                args.qmax,
            ) = params.tolist()
        else:
            args.ma, args.shift, args.za, args.zo, args.qmin, args.qmax = (
                params.tolist()
            )
        args.operation = operation
        original = source.clone()
        self.lib.pulp_cluster_integer_entry(ctypes.byref(args))
        torch.testing.assert_close(source, original, rtol=0, atol=0)
        self.assertTrue(torch.all(storage[:4] == 123))
        self.assertTrue(torch.all(storage[-4:] == 123))
        self.assertLessEqual(ctypes.c_size_t.in_dll(self.lib, "host_peak").value, 512)
        return args.status, output

    def test_add_all_byte_pairs_scales_zero_points_clipping_and_alpha(self):
        a = torch.arange(256, dtype=torch.uint8).repeat_interleave(256)
        b = torch.arange(256, dtype=torch.uint8).repeat(256)
        for ratios, zero_points in (
            ((Fraction(1, 2), Fraction(3, 4)), (17, 129, 64)),
            ((Fraction(3, 7), Fraction(-2, 9)), (255, 128, 0)),
            ((Fraction(1, 2), Fraction(1, 2)), (0, 0, 0)),
        ):
            multipliers, shift = fixed_point(ratios)
            params = torch.tensor([*multipliers, shift, *zero_points, 0, 255])
            status, result = self.run_worker(a, params, 1, b)
            self.assertEqual(status, 0)
            expected = torch.ops.pulp.quantized_add(a, b, params)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            real = (
                (
                    (a.double() - zero_points[0]) * float(ratios[0])
                    + (b.double() - zero_points[1]) * float(ratios[1])
                )
                .round()
                .add(zero_points[2])
                .clamp(0, 255)
                .byte()
            )
            self.assertLessEqual((result.int() - real.int()).abs().max().item(), 1)

    def test_requantize_tails_retry_and_failure(self):
        multipliers, shift = fixed_point([Fraction(7, 11)])
        params = torch.tensor([*multipliers, shift, 127, 50, 10, 200])
        for count in (0, 1, 3, 7, 257, 65537):
            source = torch.arange(count).byte()
            status, result = self.run_worker(source, params, 0, limit=280)
            self.assertEqual(status, 0)
            torch.testing.assert_close(
                result,
                torch.ops.pulp.requantize_per_tensor(source, params),
                rtol=0,
                atol=0,
            )
        status, _ = self.run_worker(
            torch.ones(12, dtype=torch.uint8), params, 0, limit=1
        )
        self.assertEqual(status, -1)

    def test_pool_channel_spatial_tiles_layouts_batches_and_ties(self):
        for shape in ((2, 13, 1, 1), (2, 17, 3, 7), (1, 9, 1, 65537), (1, 1, 1, 2)):
            for channels_last in (False, True):
                source = torch.arange(torch.tensor(shape).prod()).reshape(shape).byte()
                if channels_last:
                    source = source.contiguous(memory_format=torch.channels_last)
                multipliers, shift = fixed_point(
                    [Fraction(7, 11)], 255 * shape[2] * shape[3]
                )
                params = torch.tensor([*multipliers, shift, 127, 77, 0, 255])
                status, result = self.run_worker(source, params, 2, limit=300)
                self.assertEqual(status, 0)
                expected = torch.ops.pulp.quantized_global_avg_pool2d(source, params)
                torch.testing.assert_close(
                    result.reshape(expected.shape), expected, rtol=0, atol=0
                )
        params = torch.tensor([1 << 46, 46, 0, 1, 0, 255])
        source = torch.tensor([[[[0, 1]]]], dtype=torch.uint8)
        status, result = self.run_worker(source, params, 2)
        self.assertEqual(status, 0)
        self.assertEqual(result.tolist(), [1])  # Round centered mean before adding zp.

    def test_pool_denominators_beyond_signed_and_unsigned_int64(self):
        for spatial, multiplier, expected in (
            (200000, 170000000000, 78),
            (300000, 100000000000, 77),
        ):
            source = torch.full((1, 1, 1, spatial), 255, dtype=torch.uint8)
            params = torch.tensor([multiplier, 46, 0, 77, 0, 255])
            status, result = self.run_worker(source, params, 2)
            self.assertEqual(status, 0)
            self.assertEqual(result.tolist(), [expected])
            self.assertEqual(
                torch.ops.pulp.quantized_global_avg_pool2d(source, params).item(),
                expected,
            )

    def test_unsafe_coefficients_are_rejected(self):
        with self.assertRaises(ValueError):
            fixed_point([Fraction(10**30)])
        params = torch.tensor([(1 << 63) - 1, 0, 8, 0, 0, 0, 0, 255])
        with self.assertRaises(ValueError):
            torch.ops.pulp.quantized_add(
                torch.ones(1, dtype=torch.uint8),
                torch.ones(1, dtype=torch.uint8),
                params,
            )


if __name__ == "__main__":
    unittest.main()
