---
tier: on-demand
read_when: profiling shows the symptom below
---
# CUDA Graph around TensorRT

- Category: host
- Applies to: fixed-shape TensorRT engines called per request or per tile, where enqueue overhead is visible

## Symptom
Short engines (a few ms) with CPU enqueue time between them; the GPU is busy only part of the wall time.

## Technique
Allocate static input and output buffers, capture `execute_async_v3` once into a CUDA Graph, then replay it. Inputs are copied into the static buffers. See `perf-torch-cuda-graphs`.

## Effect
0.13 to 0.42 ms per call; lossless. Worth more combined with fused kernels and FP16.

## Side effects
Static buffers and shapes; one graph per shape. The outputs live in the graph's buffers, so copy them out before the next replay.

## Does not apply
- Engines with dynamic shapes per call.

## Evidence
- 2026-09: image-to-guidance-map CNN service, B1 RoI 1024x1024, RTX 5090, TensorRT 11.3: FP32 engine one RoI -0.15 ms (2k) to -0.42 ms (two RoIs), L0; with an FP16 engine -0.13 to -0.31 ms.
