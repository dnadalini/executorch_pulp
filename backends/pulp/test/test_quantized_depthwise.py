# Copyright (c) 2026 KU Leuven.
# All rights reserved.

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


class TestPulpQuantizedDepthwise(unittest.TestCase):
    def test_reference_layout_bias_padding_stride_and_resize(self):
        generator = torch.Generator().manual_seed(31)
        out = torch.empty(
            1, 5, 1, 1, dtype=torch.uint8, memory_format=torch.channels_last
        )
        for h, w, kh, kw, stride, padding in (
            (3, 7, 1, 1, [1, 1], [0, 0]),
            (7, 5, 3, 3, [2, 1], [1, 1]),
            (4, 8, 2, 3, [1, 2], [2, 1]),
        ):
            for with_bias in (False, True):
                raw = torch.randint(
                    0, 256, (1, 5, h, w), dtype=torch.uint8, generator=generator
                ).contiguous(memory_format=torch.channels_last)
                weight = torch.randint(
                    -8, 9, (5, 1, kh, kw), dtype=torch.int8, generator=generator
                )
                bias = (
                    torch.tensor([100000, -100000, 513, -257, 0], dtype=torch.int32)
                    if with_bias
                    else None
                )
                accumulator = F.conv2d(
                    raw.double(),
                    weight.double(),
                    stride=stride,
                    padding=padding,
                    groups=5,
                ).to(torch.int64)
                if bias is not None:
                    accumulator += bias.to(torch.int64).reshape(1, -1, 1, 1)
                expected = (accumulator >> 3).clamp(0, 255).to(torch.uint8)
                args = (
                    raw,
                    weight,
                    bias,
                    torch.tensor([1.0, 1.0, 8.0]),
                    torch.zeros(3, dtype=torch.int32),
                    stride,
                    padding,
                )
                actual = torch.ops.pulp.quantized_depthwise_conv2d(*args)
                torch.ops.pulp.quantized_depthwise_conv2d.out(*args, out=out)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(out, expected)
                self.assertTrue(actual.is_contiguous(memory_format=torch.channels_last))
                self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))

    def test_multiplier_two_is_rejected_by_quantizer(self):
        model = torch.nn.Conv2d(3, 6, 3, padding=1, groups=3).eval()
        example = torch.rand(1, 3, 4, 5).contiguous(memory_format=torch.channels_last)
        exported = torch.export.export(model, (example,))
        prepared = prepare_pt2e(exported.module(), PulpQuantizer())
        prepared(example)
        converted = convert_pt2e(prepared)
        self.assertFalse(
            any("quantize_per_tensor" in str(n.target) for n in converted.graph.nodes)
        )

    def test_lowering_oihw_dynamic_shapes_and_serialization(self):
        torch.manual_seed(3)
        model = torch.nn.Conv2d(5, 5, (2, 3), padding=(1, 1), groups=5).eval()
        example = torch.rand(1, 5, 5, 7).contiguous(memory_format=torch.channels_last)
        spatial = {
            2: torch.export.Dim("height", min=3, max=8),
            3: torch.export.Dim("width", min=3, max=9),
        }
        exported = torch.export.export(model, (example,), dynamic_shapes=(spatial,))
        prepared = prepare_pt2e(exported.module(), PulpQuantizer())
        prepared(example)
        converted = convert_pt2e(prepared)
        edge = to_edge(
            torch.export.export(converted, (example,), dynamic_shapes=(spatial,)),
            compile_config=pulp_edge_compile_config(),
        )
        edge = edge.transform(PulpPassManager())
        nodes = [
            n
            for n in edge.exported_program().graph.nodes
            if "pulp.quantized_depthwise_conv2d" in str(n.target)
        ]
        self.assertEqual(len(nodes), 1)
        weight = nodes[0].args[1].meta["val"]
        self.assertEqual(tuple(weight.shape), (5, 1, 2, 3))
        reference = edge.exported_program().module()
        state = edge.exported_program().state_dict
        buffers = edge.exported_program().graph_signature.inputs_to_buffers
        conv = nodes[0]
        packed_weight = state[buffers[conv.args[1].name]]
        bias = state[buffers[conv.args[2].name]]
        scales = state[buffers[conv.args[3].name]]
        quantized_input = torch.round(example * (1.0 / float(scales[0]))).clamp(0, 255)
        accumulator = F.conv2d(
            quantized_input.double(), packed_weight.double(), padding=(1, 1), groups=5
        ).to(torch.int64)
        accumulator += bias.to(torch.int64).reshape(1, -1, 1, 1)
        shift = round(-torch.log2(scales[0] * scales[1] / scales[2]).item())
        expected = (accumulator >> shift).clamp(0, 255).float() * scales[2]
        torch.testing.assert_close(reference(example), expected)
        for h, w in ((8, 9), (3, 3), (5, 7)):
            result = reference(
                torch.rand(1, 5, h, w).contiguous(memory_format=torch.channels_last)
            )
            self.assertEqual(tuple(result.shape), (1, 5, h + 1, w))
        self.assertTrue(edge.to_executorch().buffer)

    @unittest.skipUnless(shutil.which("cc"), "Host C compiler is required")
    def test_native_runtime_and_kernel_against_reference(self):
        # Compile the actual runtime and kernel with host substitutes for SDK
        # scheduling and PULP intrinsics; keep the arena and indexing unchanged.
        backend = Path(__file__).resolve().parents[1]
        root = backend.parents[2]
        kernel_root = root / "preliminary_material/pulp-nn-example/pulp-nn/32bit"
        if not kernel_root.exists():
            self.skipTest("Checked-in PULP-NN source tree is required")
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            shutil.copyfile(backend / "test/dma_host_stub.h", tmp / "dma_host_stub.h")
            (tmp / "pmsis.h").write_text(
                r"""
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include "dma_host_stub.h"
extern int host_core;
extern int host_fail_alloc;
struct pi_device { int unused; };
struct pi_cluster_conf { int unused; };
struct pi_cluster_task { void (*entry)(void*); void *arg; };
static inline void pi_cluster_conf_init(struct pi_cluster_conf *c) { (void)c; }
static inline void pi_open_from_conf(struct pi_device *d, void *c) { (void)d; (void)c; }
static inline int pi_cluster_open(struct pi_device *d) { (void)d; return 0; }
static inline void pi_cluster_close(struct pi_device *d) { (void)d; }
static inline int pi_core_id(void) { return host_core; }
static inline void pi_cl_team_barrier(int n) { (void)n; }
static inline void *pi_cl_l1_malloc(int n, size_t bytes) { (void)n; return host_fail_alloc ? NULL : malloc(bytes); }
static inline void pi_cl_l1_free(int n, void *p, size_t bytes) { (void)n; (void)bytes; free(p); }
static inline void pi_cl_team_fork(int n, void (*f)(void*), void *p) { for (host_core=0; host_core<n; ++host_core) f(p); }
static inline struct pi_cluster_task *pi_cluster_task(struct pi_cluster_task *t, void (*f)(void*), void *p) { t->entry=f; t->arg=p; return t; }
static inline void pi_cluster_send_task_to_cl(struct pi_device *d, struct pi_cluster_task *t) { (void)d; t->entry(t->arg); }
"""
            )
            (tmp / "pulp_nn_utils.h").write_text(
                r"""
#pragma once
#include <stdint.h>
typedef unsigned char v4u __attribute__((vector_size(4)));
typedef signed char v4s __attribute__((vector_size(4)));
static inline int host_dot(v4u a, v4s b, int s) { for (int i=0;i<4;++i) s += a[i]*b[i]; return s; }
static inline int host_clip(int x, int bits) { (void)bits; return x<0 ? 0 : x>255 ? 255 : x; }
#define __builtin_pulp_sdotusp4 host_dot
#define __builtin_pulp_clipu_r host_clip
static inline uint8_t pulp_nn_quant_u8(int s, int m, int shift) { return host_clip((s*m)>>shift, 255); }
static inline uint8_t pulp_nn_bn_quant_u8(int s, int k, int l, int shift) { return host_clip((s*k+l)>>shift, 255); }
"""
            )
            (tmp / "host.c").write_text(
                "int host_core; int host_fail_alloc; void pulp_nn_conv_Ho_parallel_i32_bias(void) {} void pulp_nn_pointwise_Co_parallel_i32_bias(void) {} void pulp_cluster_global_pool_entry(void *p){(void)p;} void pulp_cluster_activation_entry(void *p){(void)p;} void pulp_cluster_quantization_entry(void *p) { (void)p; }\n"
            )
            library_path = tmp / "depthwise.so"
            subprocess.run(
                [
                    "cc",
                    "-shared",
                    "-fPIC",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-DNUM_CORES=8",
                    "-DPULP_NN_INT32_BIAS",
                    "-I" + str(tmp),
                    "-I" + str(kernel_root / "include"),
                    str(backend / "ops/pulp_runtime_context.c"),
                    str(kernel_root / "src/pulp_nn_depthwise_generic.c"),
                    str(tmp / "host.c"),
                    "-o",
                    str(library_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            lib = ctypes.CDLL(str(library_path))

            class Args(ctypes.Structure):
                _fields_ = (
                    [
                        (name, ctypes.c_void_p)
                        for name in ("input", "weight", "bias", "output")
                    ]
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
                    + [
                        ("scratch_bytes", ctypes.c_size_t),
                        ("status", ctypes.c_int),
                        ("activation_max_plus_one", ctypes.c_uint16),
                    ]
                )

            lib.pulp_runtime_run_depthwise_conv2d.argtypes = [ctypes.POINTER(Args)]
            self.assertEqual(lib.pulp_runtime_initialize(), 1)
            generator = torch.Generator().manual_seed(41)
            for index in range(60):
                c = (1, 3, 5, 8, 11)[index % 5]
                h, w = 2 + index % 6, 3 + index % 7
                kh, kw = 1 + index % 3, 1 + (index // 3) % 4
                ph, pw = index % 3, (index // 2) % 3
                sh, sw = 1 + index % 2, 1 + (index // 2) % 2
                oh, ow = (h + 2 * ph - kh) // sh + 1, (w + 2 * pw - kw) // sw + 1
                if min(oh, ow) <= 0:
                    continue
                raw = torch.randint(
                    0, 256, (1, c, h, w), dtype=torch.uint8, generator=generator
                ).contiguous(memory_format=torch.channels_last)
                weight = torch.randint(
                    -128, 128, (c, 1, kh, kw), dtype=torch.int8, generator=generator
                )
                bias = (
                    torch.randint(
                        -100000, 100000, (c,), dtype=torch.int32, generator=generator
                    )
                    if index % 2
                    else None
                )
                out = torch.empty(
                    1, c, oh, ow, dtype=torch.uint8, memory_format=torch.channels_last
                )
                args = Args(
                    raw.data_ptr(),
                    weight.data_ptr(),
                    bias.data_ptr() if bias is not None else None,
                    out.data_ptr(),
                    w,
                    h,
                    c,
                    ow,
                    oh,
                    c,
                    kw,
                    kh,
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
                self.assertEqual(
                    lib.pulp_runtime_run_depthwise_conv2d(ctypes.byref(args)), 1
                )
                expected = F.conv2d(
                    raw.double(),
                    weight.double(),
                    stride=(sh, sw),
                    padding=(ph, pw),
                    groups=c,
                ).to(torch.int64)
                if bias is not None:
                    expected += bias.to(torch.int64).reshape(1, -1, 1, 1)
                torch.testing.assert_close(
                    out, (expected >> 3).clamp(0, 255).to(torch.uint8)
                )
                maximum = (0, 1, 17, 255)[index % 4]
                args.activation_max_plus_one = maximum + 1
                self.assertEqual(
                    lib.pulp_runtime_run_depthwise_conv2d(ctypes.byref(args)), 1
                )
                torch.testing.assert_close(
                    out, (expected >> 3).clamp(0, maximum).to(torch.uint8)
                )
            ctypes.c_int.in_dll(lib, "host_fail_alloc").value = 1
            self.assertEqual(
                lib.pulp_runtime_run_depthwise_conv2d(ctypes.byref(args)), 0
            )
            lib.pulp_runtime_shutdown()
            self.assertEqual(
                lib.pulp_runtime_run_depthwise_conv2d(ctypes.byref(args)), 0
            )


if __name__ == "__main__":
    unittest.main()
