# Copyright (c) 2026 ETH Zurich and University of Bologna.
"""Pointwise/linear lowering and the actual output-channel-parallel C kernel."""

import ctypes
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from executorch.backends.pulp.edge_compile_config import pulp_edge_compile_config
from executorch.backends.pulp.passes.pulp_pass_manager import PulpPassManager
from executorch.backends.pulp.quantizer.quantizer import PulpQuantizer
from executorch.exir import to_edge
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class ConvArgs(ctypes.Structure):
    _fields_ = (
        [(name, ctypes.c_void_p) for name in ("input", "weight", "bias", "output")]
        + [
            (name, ctypes.c_uint16)
            for name in (
                "input_w",
                "input_h",
                "input_channels",
                "output_w",
                "output_h",
                "output_channels",
                "kernel_w",
                "kernel_h",
                "padding_top",
                "padding_bottom",
                "padding_left",
                "padding_right",
                "stride_w",
                "stride_h",
                "out_shift",
            )
        ]
        + [("scratch_bytes", ctypes.c_size_t), ("status", ctypes.c_int)]
    )


class PointwiseArgs(ctypes.Structure):
    _fields_ = [("kernel", ConvArgs), ("batches", ctypes.c_size_t)]


class TestPointwiseLowering(unittest.TestCase):
    def test_pointwise_and_linear_reference_and_serialization(self):
        torch.set_num_threads(1)
        for shape, linear, bias in (
            ((1, 5, 3, 7), False, True),
            ((5,), True, False),
            ((2, 5), True, True),
            ((2, 3, 5), True, True),
        ):
            with self.subTest(shape=shape, linear=linear, bias=bias):
                torch.manual_seed(10)
                model = (
                    torch.nn.Linear(5, 11, bias=bias)
                    if linear
                    else torch.nn.Conv2d(
                        5, 11, 1, stride=(2, 1), padding=(1, 0), bias=bias
                    )
                ).eval()
                data = torch.rand(shape)
                if not linear:
                    data = data.contiguous(memory_format=torch.channels_last)
                prepared = prepare_pt2e(
                    torch.export.export(model, (data,)).module(), PulpQuantizer()
                )
                prepared(data)
                edge = to_edge(
                    torch.export.export(convert_pt2e(prepared), (data,)),
                    compile_config=pulp_edge_compile_config(),
                ).transform(PulpPassManager())
                ep = edge.exported_program()
                nodes = [
                    n
                    for n in ep.graph.nodes
                    if "pulp.quantized_pointwise" in str(n.target)
                ]
                self.assertEqual(len(nodes), 1)
                node = nodes[0]
                self.assertEqual(node.args[7], linear)
                buffers = ep.graph_signature.inputs_to_buffers
                weights = ep.state_dict[buffers[node.args[1].name]]
                scales = ep.state_dict[buffers[node.args[3].name]]
                biases = (
                    ep.state_dict[buffers[node.args[2].name]]
                    if node.args[2] is not None
                    else None
                )
                self.assertEqual(tuple(weights.shape), (11, 5))
                q = torch.round(data * (1.0 / float(scales[0]))).clamp(0, 255)
                if linear:
                    accumulator = F.linear(q.double(), weights.double()).to(torch.int64)
                    if biases is not None:
                        accumulator += biases.to(torch.int64)
                else:
                    accumulator = F.conv2d(
                        q.double(),
                        weights.double().reshape(11, 5, 1, 1),
                        stride=(2, 1),
                        padding=(1, 0),
                    ).to(torch.int64)
                    if biases is not None:
                        accumulator += biases.to(torch.int64).reshape(1, -1, 1, 1)
                shift = round(-torch.log2(scales[0] * scales[1] / scales[2]).item())
                expected = (accumulator >> shift).clamp(0, 255).float() * scales[2]
                torch.testing.assert_close(ep.module()(data), expected)
                self.assertTrue(edge.to_executorch().buffer)

    def test_unquantized_noncontiguous_linear_retains_float_fallback(self):
        model = torch.nn.Linear(5, 7).eval()
        inputs = torch.rand(5, 2).t()
        edge = to_edge(
            torch.export.export(model, (inputs,)),
            compile_config=pulp_edge_compile_config(),
        ).transform(PulpPassManager())
        torch.testing.assert_close(
            edge.exported_program().module()(inputs), model(inputs)
        )
        self.assertFalse(
            any(
                "pulp.quantized_pointwise" in str(n.target)
                for n in edge.exported_program().graph.nodes
            )
        )
        self.assertTrue(edge.to_executorch().buffer)

    def test_dynamic_linear_rows_and_out_resize(self):
        torch.manual_seed(2)
        data = torch.rand(2, 5)
        spec = ({0: torch.export.Dim("rows", min=1, max=5)},)
        prepared = prepare_pt2e(
            torch.export.export(
                torch.nn.Linear(5, 7).eval(), (data,), dynamic_shapes=spec
            ).module(),
            PulpQuantizer(),
        )
        prepared(data)
        edge = to_edge(
            torch.export.export(convert_pt2e(prepared), (data,), dynamic_shapes=spec),
            compile_config=pulp_edge_compile_config(),
        ).transform(PulpPassManager())
        for rows in (5, 1, 3):
            self.assertEqual(
                tuple(edge.exported_program().module()(torch.rand(rows, 5)).shape),
                (rows, 7),
            )
        self.assertTrue(edge.to_executorch().buffer)
        out = torch.empty(0, dtype=torch.uint8)
        weight = torch.ones(7, 5, dtype=torch.int8)
        for rows in (5, 1, 3):
            inputs = torch.ones(rows, 5, dtype=torch.uint8)
            torch.ops.pulp.quantized_pointwise.out(
                inputs,
                weight,
                None,
                torch.tensor([1.0, 1.0, 1.0]),
                torch.zeros(3, dtype=torch.int32),
                [1, 1],
                [0, 0],
                True,
                out=out,
            )
            self.assertEqual(tuple(out.shape), (rows, 7))
            self.assertTrue(torch.all(out == 5))


@unittest.skipUnless(shutil.which("cc"), "Host C compiler required")
class TestPointwiseNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        tmp = Path(cls.directory.name)
        backend = Path(__file__).resolve().parents[1]
        nn = backend.parents[2] / "preliminary_material/pulp-nn-example/pulp-nn/32bit"
        (tmp / "pmsis.h").write_text(
            r"""
#pragma once
#include <stdint.h>
#include <stdlib.h>
extern int host_core,host_fail;
struct pi_device {int unused;}; struct pi_cluster_conf {int unused;};
struct pi_cluster_task {void (*entry)(void*);void* arg;};
static inline void pi_cluster_conf_init(struct pi_cluster_conf*c){(void)c;}
static inline void pi_open_from_conf(struct pi_device*d,void*c){(void)d;(void)c;}
static inline int pi_cluster_open(struct pi_device*d){(void)d;return 0;}
static inline void pi_cluster_close(struct pi_device*d){(void)d;}
static inline int pi_core_id(void){return host_core;}
static inline void pi_cl_team_barrier(int n){(void)n;}
static inline void* pi_cl_l1_malloc(int n,size_t s){(void)n;return host_fail?NULL:malloc(s);}
static inline void pi_cl_l1_free(int n,void*p,size_t s){(void)n;(void)s;free(p);}
static inline void pi_cl_team_fork(int n,void(*f)(void*),void*p){for(host_core=0;host_core<n;++host_core)f(p);}
static inline struct pi_cluster_task* pi_cluster_task(struct pi_cluster_task*t,void(*f)(void*),void*p){t->entry=f;t->arg=p;return t;}
static inline void pi_cluster_send_task_to_cl(struct pi_device*d,struct pi_cluster_task*t){(void)d;t->entry(t->arg);}
"""
        )
        (tmp / "pulp_nn_utils.h").write_text(
            r"""
#pragma once
#include <stdint.h>
typedef unsigned char v4u __attribute__((vector_size(4)));
typedef signed char v4s __attribute__((vector_size(4)));
static inline int host_dot(v4u a,v4s b,int s){for(int i=0;i<4;++i)s+=a[i]*b[i];return s;}
static inline int host_clip(int x,int n){(void)n;return x<0?0:x>255?255:x;}
#define __builtin_pulp_sdotusp4 host_dot
#define __builtin_pulp_clipu_r host_clip
static inline uint8_t pulp_nn_quant_u8(int s,int m,int shift){return host_clip((s*m)>>shift,255);}
static inline uint8_t pulp_nn_bn_quant_u8(int s,int k,int l,int shift){return host_clip((s*k+l)>>shift,255);}
"""
        )
        (tmp / "host.c").write_text(
            "int host_core,host_fail; void pulp_nn_conv_Ho_parallel_i32_bias(void){} void pulp_nn_depthwise_generic_i32_bias(void){} void pulp_cluster_quantization_entry(void*p){(void)p;}\n"
        )
        library = tmp / "pointwise.so"
        subprocess.run(
            [
                "cc",
                "-shared",
                "-fPIC",
                "-O2",
                "-fsanitize=undefined",
                "-fno-sanitize-recover=undefined",
                "-DNUM_CORES=8",
                "-DPULP_NN_INT32_BIAS",
                "-I" + str(tmp),
                "-I" + str(nn / "include"),
                str(backend / "ops/pulp_runtime_context.c"),
                str(nn / "src/pulp_nn_pointwise_Co_parallel.c"),
                str(nn / "src/pulp_nn_matmul.c"),
                str(tmp / "host.c"),
                "-o",
                str(library),
            ],
            check=True,
        )
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.pulp_runtime_run_pointwise.argtypes = [ctypes.POINTER(PointwiseArgs)]
        cls.lib.pulp_runtime_initialize()

    @classmethod
    def tearDownClass(cls):
        cls.lib.pulp_runtime_shutdown()
        cls.directory.cleanup()

    def test_channel_partition_bias_odd_tails_and_spatial_parameters(self):
        torch.set_num_threads(1)
        gen = torch.Generator().manual_seed(41)
        for index in range(96):
            ci = (1, 3, 4, 5, 8, 13)[index % 6]
            co = (1, 3, 7, 8, 11, 17, 32, 35)[(index // 6) % 8]
            linear = index % 4 == 0
            batches = (1, 2, 5)[index % 3] if linear else 1
            h, w = (1, 1) if linear else (2 + index % 4, 3 + (index // 2) % 4)
            sh, sw = (1, 1) if linear else (1 + index % 2, 1 + (index // 2) % 2)
            ph, pw = (0, 0) if linear else (index % 2, (index // 3) % 2)
            oh, ow = (h + 2 * ph - 1) // sh + 1, (w + 2 * pw - 1) // sw + 1
            raw = torch.randint(
                0, 32, (batches, ci, h, w), dtype=torch.uint8, generator=gen
            ).contiguous(memory_format=torch.channels_last)
            weight = torch.randint(-4, 5, (co, ci), dtype=torch.int8, generator=gen)
            bias = (
                (torch.arange(co, dtype=torch.int32) * 37 - 250) if index % 3 else None
            )
            out = torch.empty(
                batches,
                co,
                oh,
                ow,
                dtype=torch.uint8,
                memory_format=torch.channels_last,
            )
            kernel = ConvArgs(
                raw.data_ptr(),
                weight.data_ptr(),
                bias.data_ptr() if bias is not None else None,
                out.data_ptr(),
                w,
                h,
                ci,
                ow,
                oh,
                co,
                1,
                1,
                ph,
                ph,
                pw,
                pw,
                sw,
                sh,
                3,
                0,
                -1,
            )
            args = PointwiseArgs(kernel, batches)
            self.assertEqual(self.lib.pulp_runtime_run_pointwise(ctypes.byref(args)), 1)
            expected = F.conv2d(
                raw.double(),
                weight.double().reshape(co, ci, 1, 1),
                stride=(sh, sw),
                padding=(ph, pw),
            ).to(torch.int64)
            if bias is not None:
                expected += bias.to(torch.int64).reshape(1, -1, 1, 1)
            torch.testing.assert_close(
                out, (expected >> 3).clamp(0, 255).to(torch.uint8), rtol=0, atol=0
            )
        ctypes.c_int.in_dll(self.lib, "host_fail").value = 1
        self.assertEqual(self.lib.pulp_runtime_run_pointwise(ctypes.byref(args)), 0)
        ctypes.c_int.in_dll(self.lib, "host_fail").value = 0


if __name__ == "__main__":
    unittest.main()
