# PULP optimized kernels

This backend adds PULP-NN operators directly to the ExecuTorch operator
registry. It is an optimized-kernel integration, not an ExecuTorch delegate.
The ExecuTorch method runs on the fabric controller (FC); the runtime opens the
cluster once and sends every supported operator to it synchronously.

## Conv2d MVP

`pulp::quantized_conv2d` currently supports:

- static batch size 1;
- channels-last `uint8` input and output;
- constant OHWI `int8` weights;
- zero bias, dilation 1, and groups 1;
- zero point 0 for input, weight, and output;
- power-of-two requantization using PULP-NN's `out_shift`;
- the checked-in PULP-NN `32bit/` implementation only.

The quantizer chooses a conservative shift from the accumulator depth. The
lowering pass verifies that the input/weight/output scale ratio is exactly
`2^-out_shift`. The runtime passes `out_mult=1`, disables batch normalization
and ReLU, and therefore uses PULP-NN's plain `clip_u8(accumulator >> out_shift)`
path. The convolution and accumulation use the PULP-NN 32-bit path; the
`int64_t` types visible in the operator schema are ExecuTorch metadata ABI
types, not 64-bit convolution arithmetic.

The im2col scratch requirement is part of the serialized operator and is
checked again at runtime:

```text
2 * NUM_CORES * IN_CH * K_H * K_W bytes
```

For the 32x8x8 input, 32-output-channel, 3x3 example this is 4,608 bytes.
The cluster entry point allocates one L1 block for its arguments, scratch,
input, weights, and output; copies the tensors from L2, executes on eight
cores, copies the result back to L2, and frees the block. There is no tiling or
L3 access.

Padding is filled with byte zero by the unmodified PULP-NN kernel. In affine
quantization, real zero is encoded as the tensor zero point. Requiring zero
point 0 therefore makes PULP-NN's padding numerically correct. This MVP cannot
represent negative input activations; a later fused batch-normalization/ReLU
path can naturally preserve this unsigned activation contract.

## Export

From the parent repository, use the Python environment containing this fork's
ExecuTorch and torchao packages:

```bash
python tests/pulp-nn_int8_conv2d/generate_cnn_graph.py \
  --output-dir tests/pulp-nn_int8_conv2d
```

The exporter writes `model.pte`, `input.bin`, and `golden.bin`. The PTE exposes
raw quantized `uint8` model I/O and contains one PULP operator, rather than an
ATen convolution or a delegate segment.

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
resizing within the reserved element capacity. Check `logs/05_memory_plan.info`
for this distinction. For the 8-input-channel/32-output-channel, maximum 8x8
example, the activation arena is 2,560 bytes (512 input + 2,048 output), separate
from weights, runtime metadata, test vectors and the runner's arena capacities.
The Python graph still retains its symbolic dimensions and range constraints.

### Supplying input data

The PTE describes a computation; it does not contain the next inference's input
values or discover their shape from raw bytes. The application supplies both:

```cpp
TensorImpl::SizesType sizes[] = {1, 8, actual_h, actual_w};
TensorImpl::DimOrderType order[] = {0, 2, 3, 1};
TensorImpl impl(ScalarType::Byte, 4, sizes, input_data, order);
Tensor input(&impl);
// Check both calls' Error results in application code.
method->set_input(input, 0);
method->execute();
```

Logical dimensions are NCHW; the contiguous data buffer is channels-last
(NHWC bytes). For this model, supply `8 * actual_h * actual_w` quantized uint8
bytes, not float samples. These are the same raw quantized values accepted by
the previous static runner. Live float samples need the export's input scale
and zero point applied before submission. `sizes`, `order`, `impl` and the
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
