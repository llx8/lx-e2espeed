---
tier: on-demand
read_when: profiling shows the symptom below
---
# TensorRT plugin build hygiene

- Category: env
- Applies to: projects that build custom TensorRT plugins or CUDA libraries

## Symptom
A frozen engine fails to deserialize with "Cannot find plugin" after a clean build; or production binaries change after an experiment build.

## Technique
Build experiments into an explicit build directory, never the production one. Pick TensorRT headers that match the runtime version and fail the build when none match. Compare builds by behavior (final output bytes and latency), never by library hash. Keep an environment check that lists every gap at once.

## Effect
Prevents lost production binaries and silent plugin load failures.

## Side effects
None.

## Does not apply
- Projects without custom native code.

## Evidence
- 2026-09: two-network image color-transfer model at 1536x1536 with D=16 attention, RTX 5090, TensorRT 10.16: an experiment build overwrote two production plugin binaries once; headers from a newer TensorRT built cleanly with zero warnings but the plugin was not found at engine load.
