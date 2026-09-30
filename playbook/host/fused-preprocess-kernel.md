---
tier: on-demand
read_when: profiling shows the symptom below
---
# Fused preprocessing kernel

- Category: host
- Applies to: GPU preprocessing made of several small kernels per request or per RoI

## Symptom
The GPU timeline shows a chain of small kernels (warp, color swap, normalize, binarize, channel pack) with launch gaps between them.

## Technique
Write the chain as one Triton kernel that reads the source once and writes the model input layout directly.

## Effect
0.23 to 0.36 ms per RoI; lossless. A small gain that grows with the RoI count.

## Side effects
One more custom kernel to maintain.

## Does not apply
- Chains that are not launch-bound.

## Evidence
- ✓ 2026-09: image-to-guidance-map CNN service, B1 RoI 1024x1024, RTX 5090, TensorRT 11.3, 2k input, one RoI: 6.11 -> 5.84 ms, service request 29.23 -> 28.98 ms, L0, lower peak memory.
