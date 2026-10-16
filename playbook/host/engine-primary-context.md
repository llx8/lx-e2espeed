---
tier: on-demand
read_when: profiling shows the symptom below
---
# TensorRT engine in the primary context

- Category: host
- Applies to: services that run a TensorRT engine through its own CUDA context (pycuda or the driver API) beside PyTorch

## Symptom
The engine runs in a context of its own, so every hand-over between PyTorch and the engine needs a cross-context event or a synchronize, and each context holds its own memory.

## Technique
Run the engine in PyTorch's primary context on its own stream, ordered with PyTorch's stream by `wait_stream` (or events) instead of a cross-context sync; pass PyTorch tensors' pointers as the engine's buffers. Drop the extra context and its pushes and pops.

## Effect
-0.25 ms (-8.8%) on a 2.8 ms path, process GPU memory 1214 -> 786 MiB; L0.

## Side effects
The engine's lifetime now follows PyTorch's context; release order at shutdown changes.

## Does not apply
- The engine on the default stream instead of its own: +0.37 ms, a loss.

## Evidence
- ✓ 2026-10: heatmap line-segmentation CNN service, RTX 5090, TensorRT 10.16: -0.25 ms (-8.8%), -428 MiB GPU memory, L0.
