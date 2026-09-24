---
tier: on-demand
read_when: profiling shows the symptom below
---
# FP16 TensorRT engine

- Category: compute
- Applies to: CNN stages running as FP32 TensorRT engines, when the user accepts an L2 difference

## Symptom
The engine is a large share of the time and runs in FP32; the model has no layers known to overflow in FP16.

## Technique
Build the whole network in FP16 with FP16 inputs and outputs, so preprocessing writes FP16 directly. With TensorRT versions that dropped the FP16 builder flag, put the precision into the ONNX. Combine with a CUDA Graph and a fused postprocessing kernel. Needs the user's approval: it changes numeric precision.

## Effect
Engine 3.71 -> 1.65 ms (2.2x); end to end about 2.4 ms per RoI; lower peak memory. L2: up to 2 to 4 levels at the model output.

## Side effects
Differences grow after JPEG encoding (up to 10 to 15 levels in the response). Keep the FP32 engine as the offline reference.

## Does not apply
- INT8 PTQ, FP8, BF16, and 2:4 sparsity on this CNN: all L3 (37 to 176 levels) and none faster than FP16.
- Mixed precision in TensorRT 11.3: one FP16 layer silently produced 85 to 242 level errors; compare every mixed engine end to end.

## Evidence
- 2026-09: image-to-guidance-map CNN service, B1 RoI 1024x1024, RTX 5090, TensorRT 11.3: 2k one RoI 5.86 -> 3.45 ms, service request 29.12 -> 26.74 ms, peak memory -136 MiB, L2.
