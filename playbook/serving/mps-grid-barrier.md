---
tier: on-demand
read_when: profiling shows the symptom below
---
# Grid-barrier kernels under MPS

- Category: serving
- Applies to: services running several worker processes on one GPU with NVIDIA MPS and TensorRT engines

## Symptom
All workers hang in `cudaStreamSynchronize`, GPU-Util 100% at very low power, forever; a timeout kill then brings the MPS server down (error 807). A hang that is happening now is read with card `mps-hang-forensics`.

## Technique
Before delivering a multi-worker service, check the ONNX for GroupNormalization and InstanceNormalization, scan every engine for grid-barrier kernels, and replace the normalization that produced them with a plugin. Prove the fix with one worker under a small MPS partition, which fails at once instead of hanging by chance.
Scan: `__myl_*` kernel names are plain text in a plan, so grep them for Inor, Gnor, ReluGnor, or MeanMean (seconds); confirm by carving the matching cubins (ELF blobs) out of the plan and reading `cuobjdump -sass` for a function with both YIELD and LDG.E.STRONG.GPU.
Partition test: a private daemon with short `CUDA_MPS_PIPE_DIRECTORY` and `CUDA_MPS_LOG_DIRECTORY` paths, one worker under `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=5`; the old plan fails its first call with CUDA error 720 or hangs, the fixed plan runs.

## Effect
Hang removed; latency unchanged.

## Side effects
None beyond the plugin.

## Does not apply
- Single-process services, or no MPS: the kernels are safe when they own the GPU.
- MPS thread-percentage partitioning in production: makes it worse unless engines are built under the same partition.
- A load test as proof: 12 workers for 600 s without a partition hung neither the old nor the fixed plan.

## Evidence
- ✓ 2026-09: 6 workers with MPS: hang reproduced, root cause found in GPU coredumps, fixed with a GroupNorm plugin, 8 stress rounds clean.
- ✓ 2026-10: DETR-style pose service, one worker at a 5% partition: the old plan's first call failed with CUDA error 720 at `__myl_Gnor` every time; the plugin plan ran 683 calls in 60 s.
