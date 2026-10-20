---
tier: on-demand
read_when: profiling shows the symptom below
---
# Exact graph rewrites to steer TensorRT kernels

- Category: compute
- Applies to: TensorRT engines from PyTorch exports where the profile shows one op form on a slow kernel or Myelin copies around it; changes to the exporting model code only

## Symptom
A per-layer or kernel profile shows a slow kernel for one op form while its mirror runs fast (k x 1 depthwise strip convolutions on a direct grouped kernel, their 1 x k partners on a fast one), Myelin materializing transposes and concatenations inside a loop, or random-init nodes (RandomUniformLike) drawn on every call.

## Technique
Rewrite the model code before export into an exactly equivalent form: a k x 1 depthwise convolution as the 1 x k one on H/W-transposed data (bit-equal in FP32); a matrix-factorization loop with its coefficients kept transposed so no x^T is needed (same math, summation order); random initial values drawn once at export with a fixed seed instead of per call (valid when they are random by design).
Check each in FP32 against the original, then time each alone, engine-only and interleaved at the real input shapes, on the production GPU before landing: TensorRT's kernel choice differs by GPU.

## Effect
RTX 5090: the three together engine 5.74 -> 4.55 ms at the main shape, end to end 16.25 -> 15.11 ms. L20, the production GPU: only the fixed random draw paid, engine -0.14 to -0.18 ms at every shape, end to end -0.13 ms; the other two lost or were even.

## Side effects
The exact rewrites move outputs at rebuild level. A fixed draw moves them once, like a new seed, and makes every build use the same values; the draw TensorRT bakes into each build landed farther from the FP32 network (100 images P50 / P90 4.81 / 14.33 px against 3.03 / 8.28 with the fixed draw; FP32 seed spread 2.75 / 7.55).

## Does not apply
- Rewrites measured on another GPU only: two that gained 0.45 to 0.49 ms on the RTX 5090 lost 0.3 to 0.5 ms or were even on the L20.

## Evidence
- ✗ 2026-10: camera-calibration CNN-transformer service, FP16 engine, TensorRT 10.16: vertical strip convolutions transposed: RTX 5090 -0.49 ms (-3.0%), L20 engine +0.32 / +0.35 ms at the two main shapes; reverted.
- ✗ 2026-10: the same, factorization coefficients kept transposed: RTX 5090 -0.45 ms (-2.9%), L20 engine +0.02 ms at the main shapes, -0.70 only at 4:1; reverted.
- ✓ 2026-10: the same, random initial bases drawn once at export instead of three per-call random nodes: L20 end to end -0.13 ms (noise 0.06), engine -0.14 to -0.18 ms at every shape; closer to FP32 than the build-baked draw (reverse test).
