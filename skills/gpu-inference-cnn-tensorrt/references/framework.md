# CNN TensorRT Framework

## Components

Maintain separate responsibilities for export, graph rewrite, engine build, artifact identity, runtime execution, shared device memory, tiling, and validation. A project may wrap them behind one API, but failures must identify the responsible layer.

## Shape policy

- Prefer static batch 1 and static spatial shapes for repeated restoration and upscale tiles.
- Use a small engine set or shape buckets when traffic has a few dominant shapes.
- Use a dynamic profile only when its build time, tactics, memory, and latency are verified across min/opt/max.
- Reject runtime shapes outside the declared contract.

## Runtime policy

- Bind existing CUDA tensor addresses on the active PyTorch stream.
- Check dtype, device, shape, strides, tensor format, and inferred output shapes.
- Allocate outputs with the format and strides required by the engine.
- Share context device memory only for serialized, non-concurrent contexts.

## Tiled modules

Use a static engine for interior and boundary tiles when padding cannot change the valid output materially. Pad boundary input, execute, crop output, and reproduce the reference overlap blend. Skip empty masked tiles before execution where semantics permit.

Diffusers VAEs are not covered by that default. Their deep receptive field makes a padded short tile observably different from executing the true short shape. Follow [vae-tiling-correctness.md](vae-tiling-correctness.md).

## Validation ladder

1. Compare eager PyTorch to exported ONNX in FP32 where possible.
2. Compare ONNX/reference to TensorRT at the selected precision.
3. Compare the complete tiled or masked stage.
4. Compare final representative images and end-to-end latency.

When a ladder boundary fails, use [precision-bisection.md](precision-bisection.md). When the engine remains slow, use [nsys-analysis.md](nsys-analysis.md) before changing tactics or writing a plugin.
