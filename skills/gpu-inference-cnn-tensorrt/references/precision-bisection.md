# TensorRT Precision Bisection

## Contents

- Freeze the contract
- Identify the failing boundary
- Bisect the graph
- Diagnose the first divergent node
- Validate the repair

## Freeze the contract

Persist the exact preprocessed CUDA input before comparing backends. Freeze dtype, layout, color order, range, shape, seed, weights, padding, interpolation, and plugin versions. Compare the same named tensors, not only the rendered image.

Record max absolute error, mean absolute error, relative error, cosine similarity, non-finite count, and task-specific image metrics. Choose tolerances from the reference task; do not use one global tolerance for every layer.

## Identify the failing boundary

Compare in this order:

1. PyTorch eager FP32 against exported ONNX FP32.
2. ONNX FP32 against TensorRT FP32.
3. TensorRT FP32 against TensorRT FP16/BF16.
4. Original graph against plugin-rewritten graph at the same precision.
5. Untiled against tiled integration, including edge tiles and blend regions.

Stop at the first failing boundary. This separates export semantics, TensorRT lowering/tactics, reduced precision, plugin behavior, and integration errors.

## Bisect the graph

Use topological node order as a practical search index, while respecting branches:

1. Infer ONNX shapes and select a tensor near the middle of the suspect path.
2. Expose it with `scripts/tools/mark_onnx_outputs.py`.
3. Capture the same tensor from the reference and candidate backends into NPZ files.
4. Compare with `scripts/tools/compare_tensor_npz.py`.
5. If the midpoint matches, move the lower bound forward; otherwise move the upper bound backward.
6. Repeat until the first divergent node or smallest divergent subgraph is isolated.

For DAG branches, bisect each dependency branch feeding the first bad merge. Do not assume numeric node order alone represents execution dependence.

## Diagnose the first divergent node

- `Resize`, grid/sample, padding: coordinate transform, antialiasing, align-corners, rounding.
- normalization: reduction axes, epsilon, accumulation dtype, layout, affine weights.
- activation fusion: operation order and intermediate rounding.
- attention: scale folding, K transpose, mask/bias broadcasting, softmax axis, accumulation and quantization granularity.
- convolution: input layout, padding, tactic precision, strongly typed I/O.
- plugin: serialization fields, format contract, dynamic dimensions, stream use, bounds, workspace, non-finite output.
- tiling: pad/crop extent, latent scale, overlap weight partition, boundary shape.

Force the suspect layer or subgraph to higher precision only after isolation. A global FP32 switch is a diagnostic boundary, not the final repair.

## Validate the repair

Re-run the isolated tensor test, downstream outputs, final images, all supported shape/profile boundaries, plugin serialization/deserialization, and end-to-end performance. Confirm that the fix does not silently select a different slow tactic.
