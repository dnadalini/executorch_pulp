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
python tests/simple_conv2d/generate_cnn_graph.py \
  --output-dir tests/simple_conv2d
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
python tests/simple_conv2d/generate_cnn_graph.py \
  --output-dir tests/simple_conv2d

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

make -C tests/simple_conv2d/runner clean all run
```

The final line should report `Validation: PASS`. The runner accepts
`MODEL_DIR` and `EXECUTORCH_BUILD_DIR` overrides if the exported artifacts or
CMake build are placed elsewhere.
