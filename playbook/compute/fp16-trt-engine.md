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
Build the whole network in FP16 with FP16 inputs and outputs, so preprocessing writes FP16 directly. With TensorRT versions that dropped the FP16 builder flag, put the precision into the ONNX and build strongly typed: TensorRT 10 built an FP16 ONNX without that flag as FP32, 2.7x slower. Combine with a CUDA Graph and a fused postprocessing kernel. Needs the user's approval: it changes numeric precision.

## Effect
Engine 3.71 -> 1.65 ms (2.2x) in a CNN; end to end -62% in a detection transformer; lower peak memory. L2: up to 2 to 4 levels at a CNN's output; near-threshold detections can change, so the user judges each result.

## Side effects
Differences grow after JPEG encoding (up to 10 to 15 levels in the response). Keep the FP32 engine as the offline reference.

## Does not apply
- INT8 PTQ, FP8, BF16, and 2:4 sparsity on this CNN: all L3 (37 to 176 levels) and none faster than FP16.
- Mixed precision in TensorRT 11.3: one FP16 layer silently produced 85 to 242 level errors; compare every mixed engine end to end.
- BF16 on a detection transformer: far worse than FP16 (mean 6.37 against 1.54 px), though it accumulates in FP32.
- "FP16 with FP32 accumulation" on TensorRT 10.16 and sm_120: matrix kernels accumulate in FP16 whatever the constraints; FP32 sums exist only with BF16 or TF32. Do not offer it.
- FP8 by Q/DQ, measured only so far: a CNN classifier -1.2% at L3 (a label flip); a detection transformer -5.2%, with a person lost in 3 of 25 images.
- FP16 inside with FP32 input and output kept by casts at the graph edges: no gain on the main scenario (-2%, inside noise), still L2. The gain needs FP16 input and output through the pipeline.
- A small CNN classifier whose engine is no longer the bottleneck (after GPU decode and a CUDA Graph): -0.08 ms, not worth a 0.23 score difference; the user went back to FP32.

## Evidence
- ✓ 2026-09: image-to-guidance-map CNN service, B1 RoI 1024x1024, RTX 5090, TensorRT 11.3: 2k one RoI 5.86 -> 3.45 ms, service request 29.12 -> 26.74 ms, peak memory -136 MiB, L2.
- ✗ 2026-09, pilot rerun of the same project with FP32 edges: -0.5 ms main scenario, -1.6 to -3.4 ms elsewhere, L2 (sobel max 3 levels on 6.3% of pixels); dropped.
- ✓ 2026-10: DETR-style pose model, 2048-px images, RTX 5090, TensorRT 10.16, pure FP16 chosen by the user over FP32, mixed, and BF16: 29.90 -> 11.32 ms (-62.2%), not L0 (a crowd image 35 vs 28 persons), accepted.
- ✓ 2026-10: ViT image feature encoder, batch 1, RTX 5090, TensorRT 10.16: model step 3.4 -> 0.6 ms, end to end -2.58 ms (-19.6%), GPU memory -960 MiB, lowest cosine 0.99999095, accepted; FP32 again later cost +1.05 ms (+62.9%) and FP16 came back.
- ✗ 2026-10: image background-type CNN classifier, behind GPU decode and a CUDA Graph, the same GPU: -0.08 ms (-5.1%), score max 0.234 off; FP32 kept.
