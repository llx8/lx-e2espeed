---
tier: on-demand
read_when: profiling shows the symptom below
---
# Enqueue all regions, copy back once

- Category: transfer
- Applies to: per-RoI or per-tile inference loops that synchronize after each item

## Symptom
Each RoI runs preprocess, engine, and a device-to-host copy with a sync, so items never overlap.

## Technique
Enqueue preprocessing, inference, and output conversion for every item into its own slot, then do one device-to-host copy at the end.

## Effect
About 0.5 ms for two RoIs; lossless; grows with the item count.

## Side effects
Output slots for the maximum item count must stay allocated.

## Does not apply
- A batch-2 engine to pair RoIs: it helped an FP32 engine (-0.9 ms, L1) but was slower with FP16.
- An auxiliary stream per RoI: no gain.

## Evidence
- ✓ 2026-09: image-to-guidance-map CNN service, B1 RoI 1024x1024, RTX 5090, TensorRT 11.3, two RoIs: -0.53 ms (FP32 engine) and -0.58 ms (FP16), L0.
