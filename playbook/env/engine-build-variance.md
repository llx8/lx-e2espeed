---
tier: on-demand
read_when: profiling shows the symptom below
---
# Engine build variance

- Category: env
- Applies to: TensorRT engines rebuilt from the same ONNX, in FP16 or in FP32 with TF32

## Symptom
Builds from one ONNX differ in latency and outputs; detections near a threshold flip between builds; an A/B of a rebuilt engine mixes the change with build noise.

## Technique
Measure the spread first: N builds, one process each, every scenario through each; the largest difference is the L1 bound. Then: reuse one build's timing cache to make later builds identical; or build N times and keep the fastest that verifies; and build the FP32 reference with the TF32 flag cleared, which removes its output spread.

## Effect
A shared timing cache gave byte-identical builds in 22 s instead of 80 s. TF32 off matched PyTorch FP32 (worst 0.08 px) at 28.9 instead of 21.7 ms. The fastest of four builds: -2% to -4%.

## Side effects
A timing cache binds to its TensorRT version and GPU; the fastest build is not always the closest.

## Does not apply
- Engines from different TensorRT versions: that difference is not build noise.

## Evidence
- ✓ 2026-09: DETR-style pose model, FP32 with TF32, TensorRT 10.16: 2 of 4 builds found 30 instead of 35 persons in a crowd image (a near-tie in a top-k query selection); TF32 off matched PyTorch.
- ✓ 2026-10: the same model, FP16: builds loading one build's timing cache were byte-identical.
- ✓ 2026-10: an image classifier and a small detector, TensorRT FP32, self-iteration trial: the fastest of four builds 1.55 -> 1.52 ms and 1.36 -> 1.31 ms, not landed.
