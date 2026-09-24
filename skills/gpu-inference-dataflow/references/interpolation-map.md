# PIL and OpenCV Resize Mapping

No cross-library mapping is automatically bit-exact. Freeze coordinate transform, antialiasing, border handling, dtype/range, rounding, and output clamp before replacement.

| CPU source | GPU candidate | Required notes |
| --- | --- | --- |
| PIL `NEAREST` | torchvision `InterpolationMode.NEAREST_EXACT` or `F.interpolate(..., mode="nearest-exact")` | PyTorch documents `nearest-exact` as matching PIL and scikit-image nearest. Still validate dtype and layout. |
| OpenCV `INTER_NEAREST` | `F.interpolate(..., mode="nearest")` | PyTorch intentionally preserves OpenCV's legacy `INTER_NEAREST` behavior. |
| OpenCV `INTER_NEAREST_EXACT` | `F.interpolate(..., mode="nearest-exact")` | Validate integer rounding at boundary coordinates. |
| PIL `BILINEAR` | torchvision resize bilinear with antialias, or `F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)` | PyTorch documents Pillow agreement for downsampling with this configuration; still validate dtype conversion and rounding. |
| OpenCV `INTER_LINEAR` | `F.interpolate(..., mode="bilinear", align_corners=False)` | Usually the closest tensor mapping; test exact service shapes. |
| OpenCV `INTER_LINEAR_EXACT` | no guaranteed bit-exact PyTorch mapping | Compare before replacing precision-sensitive masks or geometry. |
| PIL `BICUBIC` | torchvision bicubic with antialias, or `F.interpolate(..., mode="bicubic", align_corners=False, antialias=True)` | PyTorch documents Pillow agreement for downsampling with this configuration. Bicubic can overshoot; clamp float output when required. |
| OpenCV `INTER_CUBIC` | `F.interpolate(..., mode="bicubic", align_corners=False)` | Treat as approximate until regression passes. |
| OpenCV `INTER_AREA` | `F.interpolate(..., mode="area")` or `adaptive_avg_pool2d` | Use for downscale only. Arbitrary ratios and integer rounding may differ. |
| PIL `BOX` | `mode="area"` for downscale | Closest common mapping, not universally bit-exact. |
| PIL `HAMMING` | no maintained direct mapping in this skill | Keep CPU or implement only for a measured hotspot with a reference test. |
| PIL `LANCZOS` | PyTorch 2.13 `F.interpolate(..., mode="lanczos", align_corners=False, antialias=True)` on CPU; validated `scripts/ops/triton_lanczos_resize.py` on GPU | Native mode is the reference implementation: Lanczos-3, 6 taps, 4D CPU only. Use the Triton candidate only after comparing service shapes against this reference. |
| OpenCV `INTER_LANCZOS4` | no equivalent bundled implementation | OpenCV uses a wider Lanczos4 filter; do not label Lanczos3 equivalent. |

## Tensor conventions

Use BCHW for PyTorch interpolation. For masks use `nearest`/`nearest-exact` unless the reference intentionally creates soft masks. For RGB float tensors state whether the range is `[0, 1]`, `[0, 255]`, or normalized. Preserve channel order when moving from OpenCV BGR or nvImageCodec output.

Benchmark transfer plus layout conversion, not only the resize operator. A GPU resize that begins after a CPU decode and ends before a CPU postprocess may not improve E2E latency.

## PyTorch 2.13 Lanczos gate

PyTorch 2.13's native Lanczos path requires a 4D tensor, CPU execution, and `antialias=True`. With `align_corners=False`, the documented target is Pillow `Image.Resampling.LANCZOS`. It is therefore useful as an offline correctness oracle, not as the optimized CUDA path.

The bundled Triton kernel uses separable Lanczos-3 and half-pixel coordinates, but configurable border mapping, intermediate precision, accumulation order, clamp, and final rounding can prevent bit equality. Run `scripts/tests/test_gpu_image_ops.py`; on a PyTorch version exposing native Lanczos, it automatically reports the GPU-versus-CPU maximum and mean absolute errors. Freeze acceptable thresholds from representative images before enabling the replacement.

`bilinear`, `bicubic`, and `lanczos` may use antialiasing in PyTorch 2.13. Do not pass `align_corners` to nearest or area modes. Bicubic and Lanczos float output can overshoot the source range; clamp only when the original CPU pipeline also clamps or converts to a saturating integer type.
