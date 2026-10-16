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
Allocate static input and output buffers, capture `execute_async_v3` once into a CUDA Graph, then replay it. Inputs are copied into the static buffers. Key each graph by input shapes and tensor addresses, and raise on an address the key does not know rather than replaying stale pointers. Capture the device-to-host readback inside the graph too, into a resident pinned buffer, when pinned memory is allowed. See `perf-torch-cuda-graphs`.

## Effect
0.07 to 1.03 ms per call (-4.2% to -14.6% end to end on 1.4 to 18 ms paths); lossless. The readback inside the graph: a further -2.6%. Worth more combined with fused kernels and FP16.

## Side effects
Static buffers and shapes; one graph per shape. The outputs live in the graph's buffers, so copy them out before the next replay. Keep every buffer the graph uses referenced for the graph's lifetime: in one service the engine's buffers were freed after capture and the replays used freed memory, which only a later change of allocations exposed (wrong masks).

## Does not apply
- Engines with dynamic shapes per call; for a few batch sizes, capture one graph per size at Init (card `dynamic-batch-profiles`).

## Evidence
- ✓ 2026-09: image-to-guidance-map CNN service, B1 RoI 1024x1024, RTX 5090, TensorRT 11.3: FP32 engine one RoI -0.15 ms (2k) to -0.42 ms (two RoIs), L0; with an FP16 engine -0.13 to -0.31 ms.
- ✓ 2026-10: DETR-style pose model, FP16, RTX 5090, TensorRT 10.16: 11.34 -> 10.31 ms (-9.1%), L0.
- ✓ 2026-10: face classification CNN service, FP16, RTX 5090, TensorRT 10.16: 2.03 -> 1.89 ms (-6.6%), P99 -2.43 ms, bitwise equal.
- ✓ 2026-10: the same service, readback inside the graph into pinned memory: 1.89 -> 1.84 ms (-2.6%), bitwise equal.
- ✓ 2026-10: image background-type CNN classifier service, RTX 5090, TensorRT 10.16, engine and score readback in the graph: -0.07 ms (-4.5%), L0.
- ✓ 2026-10: heatmap line-segmentation CNN service, the same GPU, normalize, engine, argmax, and readback in one graph: -0.37 ms (-14.6%), L0.
- ✓ 2026-10: ViT image feature encoder service, FP16, the same GPU: -0.12 ms (-6.4%), L0.
- ✓ 2026-10: two-model mask service, both engines in graphs: -0.78 ms (-4.2%), L0.
