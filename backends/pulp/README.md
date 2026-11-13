# PULP optimized kernels

This backend adds PULP-NN operators directly to the ExecuTorch operator
registry. It is an optimized-kernel integration, not an ExecuTorch delegate.
The ExecuTorch method runs on the fabric controller (FC). The runtime opens the
cluster once and sends supported convolutions and DMA-tiled quantization /
dequantization to it synchronously. Conversion arithmetic runs on the cluster
FP32 units, partitioned over the configured cores.

## Conv2d MVP

`pulp::quantized_conv2d` currently supports:

- static batch size 1;
- channels-last `uint8` input and output;
- constant OHWI `int8` weights;
- optional constant `int32` bias, dilation 1, and groups 1;
- zero point 0 for input, weight, and output;
- power-of-two requantization using PULP-NN's `out_shift`;
- the checked-in PULP-NN `32bit/` implementation with its int32-bias extension.

The lowered graph contains persistent buffers for packed weights, quantized
bias (when present), and two three-element tensors: `scales` (float32) and
`zero_points` (int32), ordered `[input, weight, output]`. They are actual
operator inputs and survive PTE serialization. Buffer names need not match
CMSIS-NN's generated names. Public model I/O is FP32. Lowering preserves boundary
conversions and emits `pulp::quantize_per_tensor_cluster` and
`pulp::dequantize_per_tensor_cluster`, surrounding the UINT8 convolution. Folded Q/DQ
pairs between supported quantized operators remain folded; weights and bias
are still quantized ahead of time.

Quantization uses round-to-nearest-even and clips to the exported UINT8 range;
dequantization computes `(q - zero_point) * scale`. Negative FP32 inputs are
accepted but saturate to zero under this backend's zero-point-zero UINT8
contract. FP32 boundaries do not expand the convolution's supported range.

PT2E derives the bias scale as `input_scale * weight_scale`, with zero point 0,
and freezes `round(float_bias / bias_scale)` as int32, one value per output
channel. Lowering rejects constants whose bias plus dot-product bounds could
overflow the int32 accumulator. Bias is added before shifting and clipping:

```text
accumulator = bias_int32[channel] + sum(input_uint8 * weight_int8)
out_shift = -log2(input_scale * weight_scale / output_scale)
output_uint8 = clamp(accumulator >> out_shift, 0, 255)
```

The runtime reads the serialized scales to determine `out_shift`, validates
that it is a power-of-two right shift in `[0, 31]`, and checks zero points.
It decodes the float32 buffers with integer arithmetic so the fabric controller
does not need floating-point instructions or floating-point underflow handling.
These parameters were previously folded into a scalar shift and discarded;
their absence from the final graph did not mean quantization was absent.
Arbitrary scale ratios, nonzero zero points, per-channel scaling, and signed
activation I/O are still outside this backend's supported quantization contract.

The original PULP-NN kernel reads each bias with `*bias++`, widening **one** int8
value per output channel; it does not reinterpret four bytes as an int32 bias.
The opt-in `PULP_NN_INT32_BIAS` build uses separately named convolution/matmul
symbols with int32 bias loads in both the paired-pixel and leftover-pixel paths.
Without the define the original int8-bias source ABI is retained.

The quantizer chooses a conservative shift from the accumulator depth. The
lowering pass verifies that the input/weight/output scale ratio is exactly
`2^-out_shift`. The runtime passes `out_mult=1`, disables batch normalization
and ReLU, and therefore uses PULP-NN's plain `clip_u8(accumulator >> out_shift)`
path, with the quantized bias initializing the accumulator. The convolution
and accumulation use the PULP-NN 32-bit path; the
`int64_t` types visible in the operator schema are ExecuTorch metadata ABI
types, not 64-bit convolution arithmetic.

The im2col scratch requirement is part of the serialized operator and is
checked again at runtime:

```text
2 * NUM_CORES * IN_CH * K_H * K_W bytes
```

For the 32x8x8 input, 32-output-channel, 3x3 example this is 4,608 bytes.
The cluster entry point allocates one L1 block for its arguments, scratch,
input, weights, aligned int32 bias, and output; copies the tensors from L2,
executes on eight cores, copies the result back to L2, and frees the block.
There is no tiling or L3 access.

Padding is filled with byte zero by the PULP-NN padding path. In affine
quantization, real zero is encoded as the tensor zero point. Requiring zero
point 0 therefore makes PULP-NN's padding numerically correct. This MVP cannot
represent negative input activations; a later fused batch-normalization/ReLU
path can naturally preserve this unsigned activation contract.

## Export across singleton spatial dimensions

Host CPU convolution's fake implementation can choose different output strides
when a downsampled spatial dimension changes from one to two. For example,
MobileNet input dimensions in `[16,32]` produce an intermediate dimension
`ceil(input / 16)` in `[1,2]`. An export traced at 32 can otherwise introduce
layout guards requiring inputs of at least 17, even though the model supports 16.

For these ranges, fold eval batchnorm, prepare channels-last input/weights,
and use the PULP export helpers:

```python
from executorch.backends.pulp.export import (
    prepare_for_pulp_export,
    pulp_export_context,
)

model = prepare_for_pulp_export(model.eval().to(memory_format=torch.channels_last))
with pulp_export_context():
    # Keep this scope around export, PT2E preparation/conversion, re-export,
    # to_edge, and PulpPassManager.
    exported = torch.export.export(model, (example_input,), dynamic_shapes=shapes)
    # ... existing quantization and PULP lowering ...
```

The helper wraps ordinary zero-padding `nn.Conv2d` modules with the temporary
`pulp::conv2d_export` operator. Its eager reference performs the same convolution;
its fake kernel computes channels-last strides explicitly, independent of the
host CPU's backend selection. Standard/depthwise/pointwise quantization and
activation fusion recognize it and lower it to the existing runtime kernels.
An unquantized remainder is translated back to ordinary ATen convolution.
The temporary operator has no device registration and must not survive lowering.
Convolution subclasses, string padding, and nonzero padding modes retain their
existing implementations.

`pulp_export_context()` enables PyTorch's scoped size-oblivious symbolic reasoning
to avoid unnecessary singleton stride guards in elementwise operators too. It
restores the previous setting on exit. It does not disable shape constraints:
out-of-range inputs are still rejected. The MobileNet generator uses both
helpers and retains the complete requested dynamic range.

## Activation fusion and cluster fallback

PT2E recognizes single-consumer convolution/linear followed by ReLU, ReLU6,
or `hardtanh(min_val=0, max_val>0)`. This covers standard, depthwise, and
pointwise convolution, including MobileNet's `hardtanh(0, 6)` representation
of ReLU6. The output quantizer belongs to the complete affine/activation
pattern. Regenerate the PTE with `PulpQuantizer` and `PulpPassManager` to enable
these patterns.

ReLU needs no additional computation: the existing UINT8, zero-point-zero
convolution output already saturates negative accumulators to zero. Bounded
activations select `pulp::quantized_conv2d_clamp`,
`pulp::quantized_depthwise_conv2d_clamp`, or
`pulp::quantized_pointwise_clamp`. The backend applies the quantized upper
bound in parallel to the L1 output before the existing L2 copy, within the
same cluster dispatch and allocation. PULP-NN sources, flags, and requantization
are unchanged. The bound is `clamp(round_even(max_val / output_scale), 0, 255)`;
like other quantized values, a dequantized ReLU6 cap follows the output scale's
grid and need not be exactly 6. Older, unfused operator schemas are retained.

Fusion is skipped when the affine output is shared or the activation is not
a supported unsigned clamp. Remaining FP32 ReLU, ReLU6/hardtanh (including
signed bounds), and GELU lower to `pulp::activation_cluster`. Its `kind`
selects ReLU (0), hardtanh (1), erf GELU (2), or tanh GELU (3). GELU is a
standalone nonlinear operation; it is not folded into PULP-NN convolution.

The activation kernel supports FP32 or per-tensor affine UINT8 input/output,
preserving physical tensor order. Lowering absorbs an input dequantizer and
an exclusive output quantizer when present, so dequantization, activation,
and requantization share one cluster invocation without an intermediate L2
FP32 tensor. Shared conversion nodes remain available to other consumers.
Quantization retains ties-to-even rounding, clipping, and the existing UINT8
parameter checks; FP32 activations preserve negative outputs when applicable.
GELU uses FP32 `erff`/`tanhf` from libm; normal floating-point roundoff can
differ from the host PyTorch implementation.

Every standalone activation owns its DMA staging and L1 arena. The configurable
`PULP_ACTIVATION_L1_BYTES` budget defaults to 32768 bytes (minimum 256), including
the tile descriptor and input/output buffers. Each element needs 2, 5, or 8
buffer bytes depending on the UINT8/FP32 input/output combination. Capacity is
rounded down to four elements, capped at 16380 elements for DMA command limits,
and reduced on allocation failure. Tiles run sequentially; each tile is DMA'd
to L1, divided among cluster cores, and DMA'd back to L2. GELU/activation
arithmetic is never dispatched on the FC.

Batchnorm folding and standalone batchnorm are deferred. Eval-mode batchnorm
with fixed statistics can be folded into constant weights/bias before
quantization, as the MobileNet test already does; training batchnorm cannot
use that transformation. A remaining batchnorm between convolution and
activation prevents direct fusion in this pass.

## Channel-tiled global average pooling

`GlobalPoolToPulpPass` replaces FP32 rank-four `aten.mean.dim` over logical
spatial dimensions `(2, 3)` (including equivalent negative axes) with
`pulp::global_avg_pool2d`. Global `adaptive_avg_pool2d(..., (1, 1))` is also
supported. Other axes, dtypes, and adaptive-pooling output sizes retain their
existing operators. Pooling handles dynamic positive height/width, multiple
batches, and either `[N,C]` or `keepdim=True` `[N,C,1,1]` output.

The operator reads contiguous NCHW or channels-last NHWC input directly.
Layout-only copies immediately before the pool can be bypassed for this
consumer. Channel groups in NHWC are gathered with 2D DMA; strides too large
for the DMA hardware use bounded per-pixel transfers instead. Quantization
boundaries remain explicit: this version pools FP32 tensors, including the
dequantized outputs of PULP-NN layers.

`PULP_GLOBAL_POOL_L1_BYTES` defaults to 32768 bytes, with a minimum of 256.
For an L1 tile of `Ct` channels and `Pt` spatial positions, the arena contains
an aligned descriptor, `4 * Ct * Pt` input bytes, and `4 * Ct` accumulator/output
bytes. The initial tile maximizes the number of **complete channel planes**
that fit this budget and the DMA payload limit. Channels are distributed
across the cluster cores, with one core owning each channel's accumulation.

If even one complete spatial plane exceeds the budget, the kernel additionally
chunks the spatial dimension, retaining partial sums in L1. It initially keeps
up to `NUM_CORES` channels in this case, to use the cores in parallel. Allocation
failure retries with fewer channels, then fewer spatial positions. Every DMA
payload is bounded to 65520 bytes. Each channel tile is loaded, reduced across
all its spatial chunks, divided by the full spatial size, and written back to
L2. Batches and tiles run sequentially inside one cluster dispatch. No pooling
arithmetic or floating-point division runs on the FC.

Accumulation and normalization use FP32. Floating-point reduction order can
differ from host PyTorch, so comparisons should use FP32 tolerances rather
than require bitwise equality. The `1x1` case uses the same operator and runtime
shape handling as larger spatial inputs; it does not specialize away the
ability to perform real dynamic global pooling.

The implementation, registration, and lowering are covered by
`test_global_avg_pool2d.py`, including dynamic shapes, both layouts, channel
tails, spatial chunking, allocation retries, and serialization.

## DMA-tiled cluster quantization

New exports lower activation conversions to distinct operators:

- `pulp::quantize_per_tensor_cluster`: FP32 to UINT8;
- `pulp::dequantize_per_tensor_cluster`: UINT8 to FP32.

Both have functional and `.out` schemas with the existing scalar arguments
`(input, scale, zero_point, quant_min, quant_max, dtype)`. The original
`pulp::quantize_per_tensor` / `pulp::dequantize_per_tensor` FC implementations
remain registered for older PTEs. Regenerate models to select the new kernels;
the deployment scripts already do this before building and running.

The numerical contract remains per-tensor affine UINT8, FP32 input/output,
finite positive FP32-representable scales, and
`0 <= quant_min <= zero_point <= quant_max <= 255`. Quantization computes
`clamp(round_to_nearest_even(input * float(1.0 / scale)) + zero_point)`.
Dequantization computes `(int(input) - zero_point) * float(scale)` without
clamping the byte input. Infinity saturates on quantization; a NaN scaled value
fails with `InvalidArgument`, as in the existing native FC kernel. Weight
quantization is still performed ahead of time; runtime INT8 activation
quantization is not added by this change. Dense tensor shape and physical
storage order are preserved, including channels-last activations.

Each operator sends one task to the cluster. The cluster controller allocates
one bounded L1 arena, then processes L2 data in sequential tiles:

1. DMA input from L2 to L1 and wait for completion.
2. Fork the conversion worker across `NUM_CORES`, with disjoint element ranges.
3. DMA the output back to L2 and wait before reusing the buffers.

There is no per-tile FC dispatch. Each tile uses one FP32 buffer and one byte
buffer (five bytes per element), plus the job descriptor and per-core error
flags. `PULP_QUANTIZATION_L1_BYTES` is a CMake cache setting, default 16384 bytes,
which bounds the complete arena. If allocation fails, the controller halves
the tile capacity and retries down to four elements. If that also fails, the
operator reports an execution error. DMA commands are capped below 64 KiB,
and partial byte tails transfer exactly their valid length. Empty tensors need
no DMA or worker launch. This implementation uses sequential tiles, with no
double buffering or overlap between DMA and computation.

Only `ops/pulp_quantization_cluster.c` performs hardware FP32 arithmetic;
wrappers retain the FC software-float ISA. The cluster source disables
fast-math and FP contraction to retain the existing rounding behavior. The
cluster must have FP32 support and use the supplied Zfinx-capable toolchain.
A double-precision reciprocal is computed once in the FC wrapper, then passed
as FP32 to the quantization workers, matching the legacy kernel.

Run the reference, lowering and host DMA-worker regressions with:

```bash
python -m unittest executorch.backends.pulp.test.test_quantization_cluster \
  executorch.backends.pulp.test.test_quantized_conv2d \
  executorch.backends.pulp.test.test_quantized_depthwise
```

The host worker tests force a 256-byte arena to exercise many tiles, odd tails,
nonzero zero points, halfway rounding, infinities, NaN rejection, allocation
failure and retries with smaller tiles. The deployment tests exercise the real
SDK DMA and cluster cores; timing results remain in each `logs/performance.md`.

## Depthwise Conv2d

`pulp::quantized_depthwise_conv2d` supports multiplier 1
(`groups == input_channels == output_channels`), dilation 1, batch size 1,
contiguous OIHW int8 weights, and the same quantization and int32-bias contract
as standard convolution. Single-channel groups-1 convolutions use the standard
operator. Other grouped convolutions and depth multipliers remain unsupported.

External activation tensors retain logical NCHW shapes and channels-last
storage. The cluster entry converts the HWC input to CHW planes in L1 before
calling `pulp_nn_depthwise_generic`; the kernel writes HWC output directly.
The patched generic kernel uses bounded patch gathers and four-element dot
products, including scalar tails for small or non-multiple-of-four kernels.
The opt-in int32-bias symbol is `pulp_nn_depthwise_generic_i32_bias`.

Depthwise scratch is derived at runtime from current dimensions rather than
serialized in the operator arguments:

```text
NUM_CORES * round_up(K_H * K_W, 4) bytes
```

One L1 arena contains the arguments, per-core scratch, aligned bias, CHW input,
OIHW weights, and HWC output. Weight storage is `OUT_CH * K_H * K_W` bytes.
The full arena must fit L1; tiling is not implemented. The wrapper reports an
execution failure if allocation fails. Spatial shapes can vary within the
exported bounds, and the C++ operator resizes its output on each invocation.

Run the reference, lowering/serialization, and host C runtime/kernel tests:

```bash
python -m unittest executorch.backends.pulp.test.test_quantized_depthwise \
  executorch.backends.pulp.test.test_quantized_conv2d
```

The host tests substitute SDK scheduling and PULP intrinsics; they do not
replace device or GVSoC validation.

## Pointwise convolution and linear

`pulp::quantized_pointwise` is the common UINT8/INT8 operator for dense 1x1
convolutions and quantized linear layers. It calls the checked-in
`pulp_nn_pointwise_Co_parallel_i32_bias` kernel, partitioning output channels
across the cluster cores. Larger convolution kernels retain the existing
spatial-parallel convolution operator; grouped depthwise layers retain the
depthwise operator, including depthwise 1x1 filters.

The operator takes `(input, weight, bias, scales, zero_points, stride, padding,
linear=False)`. Weights are a contiguous OI matrix in both modes. Input/output
are channels-last NCHW tensors in convolution mode. In linear mode, contiguous
inputs of any supported positive rank retain their leading dimensions and
replace the final feature dimension with the output features. The runtime
flattens those leading dimensions into a row count and executes each vector
with **input/output H=W=1**, sharing weights and bias across rows. No graph
reshape/copy operators are required for eligible linear inputs.

The PULP quantizer now annotates `aten.linear` with the same per-tensor UINT8
activation, symmetric INT8 weight, INT32 bias, zero-point-zero and power-of-two
right-shift contract as convolution. `pulp_edge_compile_config()` preserves
linear until PULP lowering consumes it. Unquantized/ineligible linear nodes are
decomposed back to float matmul/addmm and shape/layout copies for portable
fallback. Public model I/O remains FP32; quantized linear outputs have the
same nonnegative/clipping behavior as the existing PULP convolutions. This
changes the quantized classifier compared with the previous FP32 fallback;
classification accuracy needs evaluation with representative calibration.

The Co-parallel kernel retains the optimized two-pixel/four-output-channel
matmul path for aligned channel counts. Its int32-bias opt-in ABI, per-core
weight and bias offsets, odd-width handling and null BN pointers are corrected.
Odd input-channel counts use bounded vector/scalar loads. Non-unit strides and
padding gather one or two input vectors per core before dot products. The
kernel's activation/BN flags stay disabled; fusion is separate work.

Pointwise scratch is derived at runtime:

```text
2 * NUM_CORES * round_up(INPUT_CHANNELS, 4) bytes
```

One L1 arena holds the arguments, scratch, aligned bias, input, OI weights and
output. Linear rows reuse the staged weights and bias. The complete operation
must fit L1; pointwise/linear EDMA tiling is not implemented in this change.
A Co-parallel mapping is particularly useful for small spatial maps and linear
vectors. It can be slower than spatial parallelism on larger feature maps;
check the actual per-instruction timings for the selected shapes.

Regenerate PTEs and golden vectors using the existing deployment scripts to
select the new operator. Existing PTEs using `pulp::quantized_conv2d` remain
valid. The standalone pointwise, three-layer and MobileNet reports now count
and display the dedicated operator, including linear MACs. Tests include
odd/even channels and widths, stride/padding, nontrivial int32 bias, dynamic
linear row counts, multi-dimensional linear inputs, float fallback and PTE
serialization. Host-native tests exercise the actual Co kernel and matmul
under undefined-behavior sanitization.

```bash
python -m unittest executorch.backends.pulp.test.test_quantized_pointwise
```

## Export

From the parent repository, use the Python environment containing this fork's
ExecuTorch and torchao packages:

```bash
python tests/pulp-nn_int8_conv2d/generate_cnn_graph.py \
  --output-dir tests/pulp-nn_int8_conv2d
```

The exporter writes `model.pte`, `input.bin`, and `golden.bin`. The PTE exposes
FP32 model I/O and contains three PULP operators: quantize, quantized convolution,
and dequantize, rather than an ATen convolution or a delegate segment.
`logs/quantization_parameters.info` records the actual scales and zero points
used by the on-device conversions. `input.bin` and `golden.bin` contain FP32
NHWC data. The runner compares floats with a small absolute tolerance (1e-5
output quantization steps, at least 1e-6), rejecting nonfinite outputs.

The operator schema now includes bias and qparam tensors. Regenerate PTEs and
rebuild the runtime together; older PTEs use the previous operator signature.

## Native build inputs

Set `EXECUTORCH_BUILD_PULP=ON`, `PULP_NN_ROOT` to the checked-in PULP-NN tree,
and `PULP_SDK_INCLUDE_DIRS` to the semicolon-separated PMSIS include paths.
`PULP_NUM_CORES` defaults to 8. The application must call
`pulp_runtime_initialize()` once before method execution and
`pulp_runtime_shutdown()` when inference is finished. The PMSIS bridge
deliberately has a C ABI because this PULP SDK revision's headers are not
C++-clean.

The fork is stored in `Edge_Dynamic_Compiler/executorch`, which satisfies
ExecuTorch's requirement that its source directory be named `executorch`.
Configure CMake directly from that directory; no source symlink is needed.

After sourcing `setup_pulp_open.sh`, a minimal Siracusa static-library build can
be configured with:

```bash
cmake -S executorch -B build-pulp \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/executorch/backends/pulp/cmake/pulp-open-toolchain.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_PULP=ON \
  -DPULP_NN_ROOT="$PWD/preliminary_material/pulp-nn-example/pulp-nn" \
  -DEXECUTORCH_BUILD_CPUINFO=OFF \
  -DEXECUTORCH_BUILD_PTHREADPOOL=OFF \
  -DEXECUTORCH_BUILD_PORTABLE_OPS=OFF \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF
cmake --build build-pulp --target executorch_core pulp_ops_lib
```

## Siracusa GVSoC runner

The combined script exports a fresh model, configures and rebuilds the runtime
libraries, then builds and runs the firmware. It stops if any step fails,
including failures piped through `tee`, so an old PTE cannot be used after an
export error. It works from any directory and can also be sourced:

```bash
INSTRUMENT=0 bash tests/pulp-nn_int8_conv2d/generate_model_and_run_on_gvsoc.sh
```

Omit `INSTRUMENT=0` for verbose tracing. `MODEL_DIR` and
`EXECUTORCH_BUILD_DIR` can override the output and runtime build directories.
Logs include both stdout and stderr under `MODEL_DIR/logs/` (by default,
`tests/pulp-nn_int8_conv2d/logs/`). Sourcing preserves the caller's directory
and shell options.

The runner embeds the PTE, input, and golden output directly in the L2 image;
it does not use L3 or a host filesystem. From the parent repository:

```bash
source setup_pulp_open.sh
python tests/pulp-nn_int8_conv2d/generate_cnn_graph.py \
  --output-dir tests/pulp-nn_int8_conv2d

cmake -S executorch -B build-pulp \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/executorch/backends/pulp/cmake/pulp-open-toolchain.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_PULP=ON \
  -DPULP_NN_ROOT="$PWD/preliminary_material/pulp-nn-example/pulp-nn" \
  -DEXECUTORCH_BUILD_CPUINFO=OFF \
  -DEXECUTORCH_BUILD_PTHREADPOOL=OFF \
  -DEXECUTORCH_BUILD_PORTABLE_OPS=OFF \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF
cmake --build build-pulp --target executorch_core pulp_ops_lib

make -C tests/pulp-nn_int8_conv2d/runner clean all run
```

The final line should report `Validation: PASS`. The runner accepts
`MODEL_DIR` and `EXECUTORCH_BUILD_DIR` overrides if the exported artifacts or
CMake build are placed elsewhere.

## Bounded dynamic spatial dimensions

The Conv2d kernel supports varying input height and width with a fixed batch of
one, fixed input/output channel counts and constant weights. In the example,
height and width independently range from 3 to 8. This is bounded dynamism:
unbounded shapes, changing rank/batch/channels, and L1 tiling are not implemented.

Preserve the dynamic shape specification in **both** `torch.export.export`
calls: before PT2E quantization and when re-exporting the converted module.
Omitting it from the second call specializes the model to the example shape.
The custom operator's fake implementation computes its output from symbolic
input dimensions, so these symbols survive the PULP lowering passes.

The standard `ConstraintBasedSymShapeEvalPass` and upper-bound memory planner
reserve enough L2 activation storage for the largest permitted shapes. No
custom memory planner is needed for this model. The serialized tensor sizes
are the upper-bound sizes, while `shape_dynamism=DYNAMIC_BOUND` permits runtime
resizing within the reserved element capacity. Check `logs/04_memory_plan.info`
for this distinction. For the 8-input-channel/32-output-channel, maximum 8x8
example, the activation arena includes FP32 boundary tensors and UINT8
intermediates, separate from weights, runtime metadata, test vectors and the
runner's arena capacities.
The Python graph still retains its symbolic dimensions and range constraints.

### Supplying input data

The PTE describes a computation; it does not contain the next inference's input
values or discover their shape from raw bytes. The application supplies both:

```cpp
TensorImpl::SizesType sizes[] = {1, 8, actual_h, actual_w};
TensorImpl::DimOrderType order[] = {0, 2, 3, 1};
TensorImpl impl(ScalarType::Float, 4, sizes, input_data, order);
Tensor input(&impl);
// Check both calls' Error results in application code.
method->set_input(input, 0);
method->execute();
```

Logical dimensions are NCHW; the contiguous data buffer is channels-last
(NHWC bytes). For this model, supply `8 * actual_h * actual_w` FP32 samples
(`4 * 8 * actual_h * actual_w` bytes). The model applies its exported input
scale and zero point on-device and returns FP32 outputs. `sizes`, `order`, `impl` and the
backing bytes must remain alive through the calls. Construct fresh metadata
for each inference; do not overwrite sizes borrowed from `MethodMeta`.

`set_input` resizes the runtime input tensor and copies the supplied data into
its planned buffer. (With `alloc_graph_input=False`, the runtime instead shares
the caller's data.) The PULP kernel computes the current output height and width
using the convolution formula and calls `resize_tensor(out, ...)`. This updates
sizes/strides within the same planned storage; it does not allocate a new L2
activation buffer. It then passes the actual dimensions to PULP-NN. Retrieve
outputs after every execution; their shape can change and subsequent inference
reuses their backing storage.

The portable runtime enforces rank, type, layout and total element capacity,
but does not enforce all original export per-axis ranges. The example runner
checks height/width limits, batch and channel count explicitly. Production
applications should do the same before `set_input`.

L1 is managed separately by `pulp_runtime_context.c`: each convolution already
allocates a block sized for its current input, output, weights and im2col
scratch, then frees it after execution. The im2col scratch formula depends on
channels and kernel size, so it stays constant when only H/W change. The
largest shape must fit available L1 because this backend has no tiling.

### Reproducing the device test

From the parent repository:

```bash
source setup_pulp_open.sh
python tests/pulp-nn_int8_conv2d/generate_cnn_graph.py
cmake --build build-pulp --target pulp_ops_lib
CCACHE_DISABLE=1 make -C tests/pulp-nn_int8_conv2d/runner all run
```

The exporter produces one dynamic PTE and 39 input/output cases: maximum,
minimum, all 36 permitted H/W combinations, then maximum again. `input.bin`
and `golden.bin` now concatenate those cases; generated `model_io.h` records
their shapes, byte counts, offsets and allowed bounds. The runner loads the
method once, iterates the cases, and validates both output shapes and values.
`INSTRUMENT` and `PRINT_MODEL_IO` remain available in its Makefile USER SECTION.
Use `--static --output-dir /tmp/pulp-static` to generate a single fixed-shape
32x32 regression model; pass `MODEL_DIR=/tmp/pulp-static` to the same runner.
