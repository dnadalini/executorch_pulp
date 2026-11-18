# Copyright (c) 2026 KU Leuven.
"""Deployment settings must match the solver's compile-time configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PulpDynamicConfig:
    l1_bytes: int = 212000
    metadata_headroom_bytes: int = 65536
    double_buffering: bool = True
    num_cores: int = 8
    max_iterations: int = 1000
    spatial_mapping_multiple: bool = True

    def __post_init__(self):
        """Validate positive deployment budgets."""
        if (
            self.l1_bytes < 256
            or self.metadata_headroom_bytes < 64
            or self.num_cores < 1
            or self.max_iterations < 1
        ):
            raise ValueError("Invalid pulp_dynamic deployment budgets")

    def serialized(self):
        return [
            self.l1_bytes,
            self.metadata_headroom_bytes,
            int(self.double_buffering),
            self.max_iterations,
            int(self.spatial_mapping_multiple),
            self.num_cores,
        ]
