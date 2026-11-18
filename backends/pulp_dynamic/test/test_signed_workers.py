# Copyright (c) 2026 KU Leuven.
"""Actual shared C workers: signed bytes, clipping, tails and DMA tiling."""

import ctypes
import unittest
from fractions import Fraction

import torch
from executorch.backends.pulp.ops.integer_ops import fixed_point
from executorch.backends.pulp.test import (
    test_activation_cluster as afixture,
    test_integer_cluster as ifixture,
    test_quantization_cluster as qfixture,
)


class TestSignedQuantization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        qfixture.TestClusterQuantization.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_signed_quantize_dequantize_rounding_and_tiles(self):
        for n in (0, 1, 3, 7, 257, 65537):
            source = (torch.arange(n).float() % 700 - 350) * 0.0625
            byte = torch.empty(n, dtype=torch.int8)
            job = qfixture.Args(
                source.data_ptr(), byte.data_ptr(), n, 0.125, 8, -7, -128, 127, 3, -1
            )
            self.lib.pulp_cluster_quantization_entry(ctypes.byref(job))
            self.assertEqual(job.status, 0)
            torch.testing.assert_close(
                byte,
                torch.ops.pulp.quantize_per_tensor(
                    source, 0.125, -7, -128, 127, torch.int8
                ),
                rtol=0,
                atol=0,
            )
            restored = torch.empty_like(source)
            job = qfixture.Args(
                byte.data_ptr(), restored.data_ptr(), n, 0.125, 0, -7, -128, 127, 2, -1
            )
            self.lib.pulp_cluster_quantization_entry(ctypes.byref(job))
            torch.testing.assert_close(
                restored, (byte.float() + 7) * 0.125, rtol=0, atol=0
            )


class TestSignedActivation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        afixture.TestActivationWorker.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_signed_relu_relu6_gelu(self):
        source = torch.arange(65537).to(torch.int8)
        for kind in range(4):
            out = torch.empty_like(source)
            args = afixture.ActivationArgs(
                source.data_ptr(),
                out.data_ptr(),
                source.numel(),
                0,
                6 if kind == 1 else float("inf"),
                0.125,
                8,
                0,
                0,
                -128,
                127,
                kind,
                2,
                2,
                -1,
            )
            self.lib.pulp_cluster_activation_entry(ctypes.byref(args))
            self.assertEqual(args.status, 0)
            expected = torch.ops.pulp.activation_cluster(
                source,
                kind,
                0,
                6 if kind == 1 else float("inf"),
                0.125,
                0,
                0.125,
                0,
                -128,
                127,
                True,
            )
            torch.testing.assert_close(out, expected, rtol=0, atol=1 if kind > 1 else 0)


class TestSignedInteger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ifixture.TestIntegerCluster.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_job(self, source, other, params, operation):
        out = torch.empty(
            (source.shape[0], source.shape[1]) if operation == 2 else source.shape,
            dtype=torch.int8,
        )
        job = ifixture.Args()
        job.input = source.data_ptr()
        job.other = other.data_ptr() if other is not None else None
        job.output = out.data_ptr()
        job.count = source.numel()
        job.operation = operation | 4
        if operation == 1:
            job.ma, job.mb, job.shift, job.za, job.zb, job.zo, job.qmin, job.qmax = (
                params.tolist()
            )
        else:
            job.ma, job.shift, job.za, job.zo, job.qmin, job.qmax = params.tolist()
        if operation == 2:
            job.batches, job.channels = source.shape[:2]
            job.spatial = source.shape[2] * source.shape[3]
            job.channels_last = source.dim_order() == (0, 2, 3, 1)
        self.lib.pulp_cluster_integer_entry(ctypes.byref(job))
        self.assertEqual(job.status, 0)
        return out

    def test_exhaustive_signed_add_and_requantize(self):
        a = torch.arange(-128, 128).to(torch.int8).repeat_interleave(256)
        b = torch.arange(-128, 128).to(torch.int8).repeat(256)
        m, p = fixed_point([Fraction(3, 7), Fraction(-5, 11)])
        params = torch.tensor([*m, p, -17, 11, -3, -128, 127])
        torch.testing.assert_close(
            self.run_job(a, b, params, 1),
            torch.ops.pulp.quantized_add(a, b, params),
            rtol=0,
            atol=0,
        )
        params = torch.tensor([m[0], p, -17, 3, -128, 127])
        torch.testing.assert_close(
            self.run_job(a, None, params, 0),
            torch.ops.pulp.requantize_per_tensor(a, params),
            rtol=0,
            atol=0,
        )

    def test_signed_pool_both_layouts_spatial_and_channel_tiles(self):
        for layout in (torch.contiguous_format, torch.channels_last):
            data = (
                torch.arange(2 * 13 * 101 * 7)
                .to(torch.int8)
                .reshape(2, 13, 101, 7)
                .contiguous(memory_format=layout)
            )
            m, p = fixed_point([Fraction(7, 11)], 255 * 101 * 7)
            params = torch.tensor([*m, p, -19, 7, -128, 127])
            torch.testing.assert_close(
                self.run_job(data, None, params, 2),
                torch.ops.pulp.quantized_global_avg_pool2d(data, params),
                rtol=0,
                atol=0,
            )


if __name__ == "__main__":
    unittest.main()
