# Copyright (c) 2026 ETH Zurich and University of Bologna.
# All rights reserved.

import unittest

import torch
from executorch.backends.pulp.edge_compile_config import pulp_edge_compile_config
from executorch.backends.pulp.passes.pulp_pass_manager import PulpPassManager
from executorch.backends.pulp.quantizer.quantizer import PulpQuantizer
from executorch.exir import to_edge
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class _Conv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 32, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TestPulpQuantizedConv2d(unittest.TestCase):
    def test_quantize_lower_and_execute_reference(self) -> None:
        torch.manual_seed(0)
        model = _Conv().eval()
        example = torch.rand(1, 32, 8, 8).to(memory_format=torch.channels_last)

        exported = torch.export.export(model, (example,))
        prepared = prepare_pt2e(exported.module(), PulpQuantizer())
        prepared(example)
        converted = convert_pt2e(prepared)
        edge = to_edge(
            torch.export.export(converted, (example,)),
            compile_config=pulp_edge_compile_config(),
        ).transform(PulpPassManager())

        graph = edge.exported_program().graph
        pulp_nodes = [
            node for node in graph.nodes if "pulp.quantized_conv2d" in str(node.target)
        ]
        self.assertEqual(len(pulp_nodes), 1)
        self.assertEqual(pulp_nodes[0].args[-1], 2 * 8 * 32 * 3 * 3)
        self.assertFalse(
            any("aten.convolution" in str(node.target) for node in graph.nodes)
        )

        placeholders = [node for node in graph.nodes if node.op == "placeholder"]
        self.assertEqual(len(placeholders), 2)
        raw_input = torch.arange(32 * 8 * 8, dtype=torch.int32).remainder(256)
        raw_input = raw_input.to(torch.uint8).reshape(1, 32, 8, 8)
        raw_input = raw_input.contiguous(memory_format=torch.channels_last)
        output = edge.exported_program().module()(raw_input)
        self.assertEqual(output.dtype, torch.uint8)
        self.assertEqual(tuple(output.shape), (1, 32, 8, 8))
        self.assertTrue(output.is_contiguous(memory_format=torch.channels_last))


if __name__ == "__main__":
    unittest.main()
