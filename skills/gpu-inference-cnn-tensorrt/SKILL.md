---
name: gpu-inference-cnn-tensorrt
description: "Move a measured CNN, restoration, segmentation, or VAE stage to TensorRT: ONNX export, plugins, engines, tiling, precision bisection."
---

# Optimize CNN Stages with TensorRT

Treat TensorRT as a maintained subsystem, not a one-line model conversion.

## Admission gate

Proceed only when profiling identifies a material CNN-like stage, representative shapes are known, output behavior can be compared, and deployment can build or ship a compatible engine. Prefer a bounded submodule over converting the entire heterogeneous pipeline.

## Framework workflow

1. Freeze the PyTorch reference module, preprocessing contract, dtype, layout, and shape set.
2. Export ONNX and compare PyTorch versus ONNX outputs before building an engine.
3. Inspect unsupported or slow graph regions and apply a maintained graph replacement or plugin only when required.
4. Select static B1 engines for stable service shapes; use bounded profiles only for genuine variability.
5. Build an engine identity from model content, hardware capability, TensorRT/plugin versions, precision, and shape profile.
6. Accept and return CUDA tensors on the current stream without NumPy or host staging.
7. Reuse context device memory for engines that are guaranteed not to execute concurrently.
8. Use static tile pad/crop and behavior-matched blending for local CNNs. Treat Diffusers VAEs as a semantic exception until boundary equivalence is proven.
9. Validate PyTorch, ONNX, and TensorRT in sequence, then benchmark end to end.
10. Fail fast on incompatible engines, plugins, shape, dtype, strides, or device.

Use `scripts/runtime/shared_trt_runtime.py` (not run on this machine, 2026-10-12) for the shared context-memory runtime and engine framework. Read [references/framework.md](references/framework.md) before adapting it and [references/operators.md](references/operators.md) when an operator or plugin is needed.

## Bundled implementations

- `scripts/plugins/tscnn_groupnorm_silu_plugin.py`: Triton-backed TensorRT GroupNorm and GroupNorm+SiLU plugin; not run on this machine (2026-10-12).
- `scripts/plugins/replace_tscnn_groupnorm_onnx.py`: TSCNN graph matcher and plugin rewrite; not run on this machine (2026-10-12).
- `scripts/plugins/replace_seedvr_groupnorm_onnx.py`: broader SeedVR/Klein GroupNorm graph rewrite; not run on this machine (2026-10-12).
- `scripts/plugins/color_attn_d16/`: SM120 D=16 fused color-attention CUDA/TensorRT plugin source.
- `scripts/examples/vae_tensorrt.py`: static convolutional VAE tile, pad/crop, blend, self-check, and runtime integration.
- `scripts/tests/`: plugin numerical and VAE tiling semantic tests.
- `scripts/tools/`: ONNX inspection, engine inspection, nsys summaries, intermediate-output marking, and tensor comparison.

Read [references/nsys-analysis.md](references/nsys-analysis.md) for TensorRT operator timing. Read [references/precision-bisection.md](references/precision-bisection.md) whenever outputs diverge.
Read [references/vae-tiling-correctness.md](references/vae-tiling-correctness.md) before changing a Diffusers VAE tile size, boundary policy, or spatial profile.
Read [references/dependencies.md](references/dependencies.md) before building or running bundled implementations.

## Production rules

- Do not build engines in the request hot path.
- Do not silently fall back when a production configuration requires TensorRT.
- Resolve weights and engines through the project's own configuration, in the way its deployment expects (for example, paths joined onto the algorithm repo root). Never hardcode absolute paths, depend on the working directory, or point at a remote weight store directly.
- Copy required reference implementation source into the optimized project and use local imports. Never make the production service import code from a neighboring reference repository.
- Plugin and other shared libraries the service needs at startup must ship with the project, checked for target GPU, CUDA, TensorRT ABI, and load path. Hardware- and version-bound `.plan`/`.engine` files follow the project's existing artifact policy; do not confuse them with required runtime libraries.
- Do not claim a conversion win without including transfers, tile handling, and workspace in the end-to-end measurement.

## Deformable convolution and tiled VAE notes

- For a segmentation network such as BiRefNet that uses `torchvision.ops.deform_conv2d`, install and register `deform_conv2d_onnx_exporter==1.2.0` before legacy ONNX export. With PyTorch 2.9 (versions as of 2026-09-24), add `JitScalarType.HALF -> ONNX FLOAT16` to `torch.onnx.symbolic_helper.cast_pytorch_to_onnx`; version 1.2.0 otherwise raises a HALF dtype `KeyError`.
- Use a static B1 `1x3x1024x1024` engine when preprocessing always resizes to 1024 square. Validate sigmoid outputs, binary-mask disagreement, and the complete downstream mask logic rather than raw logits alone.
- For diffusion VAEs, keep batch static at B1 but preserve the reference tile geometry and exact short-edge execution. Use bounded dynamic H/W when a static engine would require padding a short tile through a non-local VAE. Never reduce the reference tile size merely to reduce workspace without final-output validation. Keep encoder and decoder identities architecture-specific because latent channels differ across Flux generations.
- Run a finite-output self-check on both VAE engines. If the FP16 decoder overflows, upcast only its mid-block attention to FP32 during export and retain FP16 for the convolutional body; do not silently accept non-finite output or promote the whole decoder without measurement.
