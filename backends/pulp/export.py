# Copyright (c) 2026 KU Leuven.
"""Give PULP convolution export a layout independent of the host CPU backend."""

from contextlib import contextmanager
from copy import copy

import executorch.backends.pulp.ops.operators  # noqa: F401

import torch
from torch.fx.experimental import _config as symbolic_config


@contextmanager
def pulp_export_context():
    """Avoid singleton stride guards throughout export, PT2E, and lowering.

    This is PyTorch's size-oblivious symbolic reasoning, not guard suppression:
    the requested shape constraints are still checked by torch.export.
    Conv2d must also use prepare_for_pulp_export to avoid host backend layout
    selection in its fake kernel. Restore the setting when leaving the scope.
    """
    with symbolic_config.patch(backed_size_oblivious=True):
        yield


class PulpConv2dForExport(torch.nn.Conv2d):
    """An export-only Conv2d with an explicit channels-last output contract."""

    def forward(self, input):
        return torch.ops.pulp.conv2d_export(
            input,
            self.weight,
            self.bias,
            list(self.stride),
            list(self.padding),
            list(self.dilation),
            self.groups,
        )


def prepare_for_pulp_export(module: torch.nn.Module) -> torch.nn.Module:
    """Wrap zero-padding Conv2d modules, sharing their parameters and training state.

    Call after eval batchnorm folding. The temporary operator must disappear in
    PulpPassManager; it has no device implementation. The original module is
    updated in place, except that a root Conv2d returns a replacement module.
    """
    if type(module) is torch.nn.Conv2d:
        if module.padding_mode != "zeros" or isinstance(module.padding, str):
            return module
        replacement = copy(module)
        replacement.__class__ = PulpConv2dForExport
        return replacement
    for name, child in list(module.named_children()):
        module.add_module(name, prepare_for_pulp_export(child))
    return module
