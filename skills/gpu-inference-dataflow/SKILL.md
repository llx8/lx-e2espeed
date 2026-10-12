---
name: gpu-inference-dataflow
description: "Move image pre/postprocessing to the GPU and cut CPU-GPU round trips: PIL/OpenCV/NumPy, JPEG, resize, blur, morphology, masks, syncs."
---

# Optimize Inference Dataflow

## Workflow

1. Annotate each stage with device, dtype, layout, shape, and representation.
2. Search the hot path for `.cpu()`, `.cuda()`, `.to()`, `.numpy()`, `.item()`, PIL/OpenCV conversions, synchronization, and repeated model construction.
3. Find repeated resize, normalize, cast, permute, mask creation, copy, and full-resolution temporary work.
4. Keep the input on GPU from the earliest worthwhile boundary through the last GPU consumer.
5. Replace CPU image operations with mature PyTorch, torchvision, Kornia, cuCIM, or nvImageCodec operations when final behavior can be validated within the product-approved tolerance; do not preserve incidental bit-level behavior at material cost.
6. Skip provably empty masks, ROIs, or tiles before invoking expensive models.
7. Change one boundary at a time and compare end-to-end latency and output behavior.
8. Preserve the project's preprocessing/postprocessing tensor contract (dtype such as FP16, range, layout, color order, interpolation, rounding) and run the complete project accuracy suite after replacements.
9. Keep crop and paste image tensors on their current CUDA device; move only compact masks or metadata to CPU when connected-components or grouping logic requires it.
10. Replace PIL resize in a GPU-resident hot path with the interpolation-map equivalent: use antialiased `torch.nn.functional.interpolate` for bicubic and vendor the bundled Triton Lanczos3 operator for Lanczos. Compare against the unchanged PIL reference on representative real sizes before delivery.
11. When a TensorRT engine has static batch 1, expose that contract on its adapter, schedule exactly one tile per call, and do not mark its batch dimension dynamic. Fail fast on a mismatched tile batch.
12. When the CPU reference warps with OpenCV, replicate its integer arithmetic bit for bit in one fused kernel that also writes the engine input, if that costs no more than a float version: the user chose the exact replica twice. See [references/opencv-warp-replica.md](references/opencv-warp-replica.md).
13. When the service decodes and the algorithm runs in another repo, widen the algorithm's input to take the CUDA tensor beside its NumPy input, so the image is never downloaded between them; the user prefers this faster path.

Read [references/patterns.md](references/patterns.md) for maintained dataflow patterns.

Read [references/interpolation-map.md](references/interpolation-map.md) before replacing PIL/OpenCV resize. Read [references/gpu-image-ops.md](references/gpu-image-ops.md) for PyTorch, Kornia, cuCIM, large-kernel, and Triton routing. Read [references/jpeg-codec.md](references/jpeg-codec.md) for CUDA JPEG boundaries and hardware decode in a service.

## Bundled implementations

- `scripts/ops/triton_lanczos_resize.py`: separable CUDA Lanczos3 for BCHW tensors.
- `scripts/ops/large_kernel_eroder.py`: rectangular pooling and Kornia shaped-kernel erosion with split passes and overlap tiles.
- `scripts/ops/large_kernel_gaussian_blur.py`: separable Gaussian blur with multi-pass sigma composition.
- `scripts/ops/cucim_distance.py`: cuCIM EDT with DLPack and PyTorch-stream interoperability.
- `scripts/ops/nvimgcodec_jpeg.py`: resident CUDA JPEG decode/encode bridge.
- `scripts/ops/opencv_warp_affine_u8.py`: bit-exact `cv2.warpAffine` (uint8, INTER_LINEAR, BORDER_CONSTANT) for OpenCV 4.x and 5.x, plain or fused with channel reversal, CHW, and a normalize lookup; `scripts/tests/test_opencv_warp_affine.py` checks it against the installed cv2.
- `scripts/tests/`: focused correctness, finite-output, latency, and peak-memory tests.

## Guardrails

- Do not migrate a cheap CPU stage merely to remove a visible transfer.
- Preserve color order, range, routing, and hard mask semantics. For soft masks, morphology, blur, resize, antialiasing, padding, and rounding, use cheaper perceptually equivalent behavior when the product explicitly allows tolerance, and validate the final output rather than demanding intermediate equality.
- Avoid custom Triton/CUDA kernels unless a measured operator remains material and no mature operator is adequate.
- Do not hide transfers inside a wrapper; make ownership explicit.
- Do not catch a GPU preprocessing, postprocessing, or codec failure and continue with PIL, OpenCV, NumPy, CPU, or another slower path in production. Raise an actionable error and terminate the request/service according to the project's error policy. This holds for inputs a hardware decoder does not take (other formats, progressive, truncated) too: they fail fast with an error response, never go to the original decoder (the user, 2026-10-12).
- Do not import operators from a reference repository at runtime. Vendor the required source into the target project, use local imports, and record its source revision.
- Do not force a full-resolution image to CPU merely because its mask grouping metadata is CPU-owned; slice and blend the image on GPU using CPU-derived integer boxes.
