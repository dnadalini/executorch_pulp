# Copyright (c) 2026 KU Leuven.
# All rights reserved.

import torch
from executorch.exir import EdgeCompileConfig


def pulp_edge_compile_config() -> EdgeCompileConfig:
    return EdgeCompileConfig(
        _check_ir_validity=False, preserve_ops=[torch.ops.aten.linear.default]
    )
