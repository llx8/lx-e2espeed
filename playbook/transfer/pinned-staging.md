---
tier: on-demand
read_when: profiling shows the symptom below
---
# Pinned staging for host-to-device copies

- Category: transfer
- Applies to: large host-to-device uploads from pageable NumPy arrays

## Symptom
Uploads of large masks or images take milliseconds; `cudaMemcpy` from pageable memory in the timeline.

## Technique
Keep a resident pinned buffer and a resident device buffer; copy into the pinned buffer (multithreaded for large arrays), then copy asynchronously, overlapping the next array's CPU copy. Check that the pinned buffer is not silently replaced by a new array (for example `np.ascontiguousarray` assigned over it).

## Effect
20 MiB upload 1.08 -> 0.39 ms; with overlap, a full-resolution upload stage 8.9 -> 1.9 ms; lossless.

## Side effects
Pinned host memory per worker; some teams rule it out for production stability.

## Does not apply
- Projects whose user ruled out pinned memory: ask at onboarding.
- Pipelines already moved to the GPU end to end, and small copies: no stable gain in one project.

## Evidence
- 2026-09: image-to-guidance-map CNN service, B1 RoI 1024x1024, RTX 5090, TensorRT 11.3, 4480x6720 masks: stage 1 8.86 -> 1.89 ms, L0; not delivered because the project ruled out pinned memory.
- 2026-09: two-network image color-transfer model at 1536x1536 with D=16 attention, RTX 5090, TensorRT 10.16: output buffer reuse, pinned memory, and a separate H2D stream gave no stable gain.
