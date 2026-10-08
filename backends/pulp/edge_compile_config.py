# Copyright (c) 2026 ETH Zurich and University of Bologna.
# All rights reserved.

from executorch.exir import EdgeCompileConfig


def pulp_edge_compile_config() -> EdgeCompileConfig:
    return EdgeCompileConfig(_check_ir_validity=False)
