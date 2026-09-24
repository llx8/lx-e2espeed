---
tier: on-demand
read_when: profiling shows the symptom below
---
# Grid-barrier kernels under MPS

- Category: serving
- Applies to: services running several worker processes on one GPU with NVIDIA MPS and TensorRT engines

## Symptom
All workers hang in `cudaStreamSynchronize`, GPU-Util 100% at very low power, forever; a timeout kill then brings the MPS server down (error 807).

## Technique
Before delivering a multi-worker service, scan every engine for grid-barrier kernels and replace the normalization that produced them with a plugin. The `gpu-inference-mps-hang` skill has the scanner, the coredump forensics, and MPS operations.

## Effect
Hang removed; latency unchanged.

## Side effects
None beyond the plugin.

## Does not apply
- Single-process services, or no MPS: the kernels are safe when they own the GPU.
- MPS thread-percentage partitioning: makes it worse unless engines are built under the same partition.

## Evidence
- 2026-09: 6 workers with MPS: hang reproduced, root cause found in GPU coredumps, fixed with a GroupNorm plugin, 8 stress rounds clean.
