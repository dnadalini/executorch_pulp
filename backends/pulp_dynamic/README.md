# Runtime-tiled PULP backend

`pulp_dynamic` exports ordinary ExecuTorch execution plans with optimized operators.
It uses the `on-device-solver` submodule to configure and execute signed INT8
standard, pointwise and multiplier-one depthwise convolutions. Linear is mapped
to pointwise with H=W=1, iterating over leading-dimension rows. This is an
operator-library integration, not a delegate.
The existing `pulp` UINT8 backend remains selectable.

## Export

```python
from executorch.backends.pulp_dynamic import (
    PulpDynamicConfig, PulpDynamicQuantizer, PulpDynamicPassManager,
)
from executorch.backends.pulp_dynamic.export import (
    prepare_for_pulp_dynamic_export, pulp_dynamic_export_context,
)
from executorch.backends.pulp.edge_compile_config import pulp_edge_compile_config
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from executorch.exir import to_edge

config = PulpDynamicConfig(l1_bytes=212000, double_buffering=True,
                           metadata_headroom_bytes=65536)
model = prepare_for_pulp_dynamic_export(model.eval())
example = example.contiguous(memory_format=torch.channels_last)
with pulp_dynamic_export_context():
    exported = torch.export.export(model, (example,), dynamic_shapes=shapes)
    prepared = prepare_pt2e(exported.module(), PulpDynamicQuantizer(config))
    for sample in calibration_inputs:
        prepared(sample)
    quantized = torch.export.export(convert_pt2e(prepared), (example,),
                                   dynamic_shapes=shapes)
    edge = to_edge(quantized, compile_config=pulp_edge_compile_config())
    edge = edge.transform(PulpDynamicPassManager(config))
    program = edge.to_executorch()
```

The helper copies the model, traces graph connectivity, folds single-consumer
eval Conv2d/BatchNorm2d pairs with fixed running statistics, and wraps convolutions
using the existing export-only channels-last operator. For models not supported
by FX symbolic tracing, fold known eval batchnorm pairs in the model and call
`prepare_for_pulp_export` directly, as the deployment generator does for
MobileNet. Training/dynamic batchnorm is not folded.

Convolutions have batch=1, dilation=1, static channel/kernel sizes, and bounded
positive spatial sizes within primitive uint16 limits. Pointwise requires
stride=1/padding=0. Depthwise multiplier is one. Other groups/configurations and
minimum tiles that cannot fit with full Ci are left portable before Q/DQ folding.
Ci reduction tiling is not available. The generator prints fallback node/reason
and final portable instruction/operator names. Applications must register the
portable operators their model needs; the example links a selected subset.

Before quantization, eligible convolutions with NCHW inputs receive an explicit
channels-last copy. Logical [N,C,H,W] dimensions stay unchanged. Copies are shared
when convolutions use the same input, and native convolution output layouts are
restored when needed by downstream views. Export memory planning includes these
buffers; unsupported groups and depthwise multipliers still remain portable.

## Numerical behavior and fusion

Activations use per-tensor symmetric signed INT8 with zero point zero, weights
signed INT8, and constant INT32 bias with scale input_scale*weight_scale.
Convolution output scaling is a derived power-of-two right shift. The reference
uses exact integer dot products followed by arithmetic shift and saturation to
[-128,127], matching the solver primitives. The compiler rejects unsafe INT32
accumulator bounds, including bias. This conservative quantization recipe needs
trained-model accuracy validation and recalibration; random-weight tests validate
execution, not accuracy or equivalence to the previous UINT8 model.

The shared `pulp::*` quantize/dequantize, activation, add, requantize and integer
pooling operators support signed INT8 and retain their UINT8 paths. Integer
activation outputs use a negative quant_min for signed storage (the new
quantizer emits the full [-128,127] range); the original activation schema has no
separate output-dtype argument. Input and output scales remain independent.
Public model I/O remains FP32. GELU and boundary conversions use cluster FP32;
byte add/requantization/global pooling use integer fixed-point arithmetic.

ReLU fuses into standard, pointwise/linear and multiplier-one depthwise kernels
using their native flag and signed zero clamp. ReLU6 and hardtanh with lower bound
zero also fuse, applying their quantized upper bound inside each L1 output tile
before DMA. Both buffering modes are supported. Shared convolution outputs and
other activation patterns remain separate nodes. Batchnorm is folded into constants before
quantization; runtime native batchnorm flags are not enabled.

Physical spatial activation storage is NHWC with logical [N,C,H,W] sizes.
Weights are OHWI for standard convolution, OI for pointwise/linear, and CHW for
depthwise. Copies needed by portable kernels are explicit; in particular portable
permute requires matching dimension orders. Dynamic view instructions use the
primary ExecuTorch primitive-op registry.

## Runtime and memory

One shared cluster runtime is initialized once per model/runner session and
closed at shutdown. Each layer submits a synchronous task. Solver calls and
compute run on the cluster; full tensors, packed weights, biases and output
buffers are in L2. There is no full-layer L1 staging in the adapter. Metadata is
generated and freed on every invocation, including repeated input shapes. Linear
rows reuse metadata only within that invocation. Execute methods serially;
concurrent model execution is not supported by this prototype runtime.

`PulpDynamicConfig` serializes tile bytes, metadata/auxiliary L2 headroom,
buffering mode, iteration setting, tile strategy and core count. The default tile
budget is 212000 bytes (decimal), not 212 KiB. The runtime checks that L1 bytes,
core count and buffering match the compiled library. Both solver and template
are compiled consistently for the selected buffering mode. The adapter requests
tiling whenever a double-buffered full layer would exceed the budget, even if a
single-buffered full layer would fit. It does not switch modes automatically.

Solver-owned metadata is forced into L2 by providing no reserved L1 metadata
allowance in the solver build. The remaining physical L1 is for stacks/runtime;
solver task stacks are 8 KiB for the master and 1 KiB per worker. Shared
operators retain their existing SDK stack defaults. The runtime
allocates initialized L2 kappa/lambda arrays because the templates transfer them
even when their arithmetic is disabled. `metadata_headroom_bytes` includes these
arrays plus the metadata table. A native allocator probe checks contiguous L2
capacity after model arena allocation, releases it immediately, and then lets
the solver allocate normally. Returned metadata size plus the auxiliary arrays
is checked against the budget and released on failure.

This is headroom budgeting, not a hard allocator partition: the solver may
allocate metadata before its size can be checked. It may terminate if allocation
fails internally. Budget model constants, planned activations, embedded test
vectors, method/temp arenas, other heap users and solver headroom against the
board's actual L2. A large L1-tiled layer still needs its full live tensors in L2.

Build-only adapters silence solver printf calls by default and route fatal
solver exits to a diagnostic containing the active logical layer id and phase.
Detectable wrapper/metadata errors fail the ExecuTorch context. Layer ids map to
serialized instruction indices in the deployment generator's layer_mapping.info.
No automatic portable recovery is attempted after partial solver execution.
Verbose solver logging perturbs latency and should be used only for diagnosis.

## Build and deployment tests

Use the existing PULP toolchain with these additional CMake settings:

```text
EXECUTORCH_BUILD_PULP_DYNAMIC=ON
PULP_DYNAMIC_SOLVER_ROOT=/path/to/on-device-solver
PULP_DYNAMIC_L1_BYTES=212000
PULP_DYNAMIC_DOUBLE_BUFFERING=ON
PULP_DYNAMIC_VERBOSE=OFF
```

The new target reuses the original PULP operator registration library and common
runtime. Link both registration libraries, `pulp_dynamic_kernels`,
`pulp_dynamic_solver`, `pulp_kernels`, `pulp_runtime_common`, `pulp_nn`, the
appropriate portable registration/kernels, and the primary ExecuTorch runtime
when dynamic primitives are present. Static archive group/whole-archive handling
is demonstrated in tests/pulp_dynamic/runner/Makefile.

From the repository root:

```bash
source tests/pulp_dynamic/generate_model_and_run_on_gvsoc.sh --model three_layers
MODEL_DIR=/tmp/pulp-dynamic-mobile source tests/pulp_dynamic/generate_model_and_run_on_gvsoc.sh --model mobilenet --min-size 16 --max-size 32
MODEL_DIR=/tmp/pulp-dynamic-large source tests/pulp_dynamic/generate_model_and_run_on_gvsoc.sh --model large_conv2d --min-size 5 --max-size 8
DOUBLE_BUFFERING=OFF MODEL_DIR=/tmp/pulp-dynamic-single source tests/pulp_dynamic/generate_model_and_run_on_gvsoc.sh --model large_conv2d --static --max-size 8
```

Models: conv2d, large_conv2d, depthwise, pointwise, linear, linear_rows,
three_layers (three convolutions plus classifier), mobilenet and portable (an
intentional stride-2 pointwise fallback). Options include --static,
--one-inference, --l1-bytes, --metadata-bytes and --single-buffering. The wrapper
keeps export/build buffering and L1 settings synchronized. Compilation and runner
settings are printed in logs. The runner method/planned arena defaults to 196608 bytes and its temporary arena
to 16384 bytes. Override `METHOD_POOL_BYTES` and `TEMP_POOL_BYTES` when larger
exported shape bounds require more space, accounting for those allocations in
the L2 budget. A tileable layer can still exceed the runner arena or physical
L2 capacity.

Default dynamic validation uses four corner shapes
and a repeated maximum shape to exercise cleanup/reconfiguration.

The large convolution has 294912 weight bytes, exceeding the default L1 budget.
Use a smaller configured L1 budget to force tiling in otherwise small layers.
Keep model export and library configuration identical. Unsupported configurations
are printed during export and portable instructions during final-plan inspection.

Profiling writes one consolidated logs/performance.md with model/instruction
averages, per-sample shapes/MACs/cycles/MAC-per-cycle, and optional runtime tiling
rows. FC timing includes configuration, allocation, DMA and cluster wait. The
separate solver/template counters are cluster cycles, not FC cycles; runtime
statistics are printed after the whole-model FC counter stops. Optional solver
statistics record logical layer ids below 256; FC instruction profiling covers
the complete execution plan. No metadata cache is used.

Host tests:

```bash
python -m unittest discover -s executorch/backends/pulp_dynamic/test
python -m unittest discover -s executorch/backends/pulp/test
PYTHONPATH=tests python -m unittest discover -s tests/pulp_profiling
```

The signed worker tests compile actual C with SDK substitutes, covering all
65536 byte pairs, independent scales/zero points, saturation, byte tails,
channel/spatial tiling, both pool layouts, activations and quantization. Native
GVSoC deployment is the validation of PMSIS allocation/DMA, not the host stubs.

## Inspecting export and lowering

The deployment generator writes the same numbered export, quantized, Edge,
backend-lowered, serialized execution-plan and detailed memory-plan logs as the
PULP-NN tests. It also writes quantization parameters and ordered per-pass
snapshots under `logs/lowering_passes/`; see the deployment-test README for the
filenames. These are export-time diagnostics and do not affect device profiling.

For another exporter, pass a read-only callback as
`PulpDynamicPassManager(config, pass_snapshot_callback=callback)`. It receives
`(stage_name, exported_program)` synchronously after each pass, plus the initial
program and finalization stages. Serialize the snapshot inside the callback,
because subsequent passes may mutate the program. Callbacks are disabled by
default. `executorch.backends.pulp_dynamic.diagnostics.ExportDiagnostics`
provides the file writer used by the deployment generator.
