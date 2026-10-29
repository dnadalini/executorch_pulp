# Copyright (c) 2026 ETH Zurich and University of Bologna.
# All rights reserved.

from executorch.exir.pass_base import ExportPass


class PulpQuantizedIOPass(ExportPass):
    """Compatibility no-op: public model I/O retains its exported FP32 contract.

    Boundary quantization is lowered by AtenToPulpPass, never erased here.
    """
