---
tier: on-demand
read_when: profiling shows the symptom below
---
# TensorRT VAE with the reference tiling

- Category: memory
- Applies to: diffusion VAEs moved to TensorRT for large images

## Symptom
A TensorRT VAE with smaller static tiles fits memory but changes the image; or an OOM appears after the switch.

## Technique
Keep the loaded VAE's tile size, overlap, traversal, and true short edge tiles; use a B1 engine with bounded dynamic height and width instead of zero-padding tiles to a static shape; route GroupNorm through a plugin. See the `gpu-inference-memory` and `gpu-inference-cnn-tensorrt` skills.

## Effect
Reconstruction error mean 5.02/255 (broken static 512 tiles) -> 0.138/255.

## Side effects
Dynamic profiles raise the context memory (3.55 GiB for one decoder); share one pool between non-concurrent contexts.

## Does not apply
- Zero-padded static tiles: padding contaminates valid pixels through a deep VAE.
- Blaming an OOM on the new engine before checking other GPU processes: one OOM came from an unrelated 16.56 GiB process.

## Evidence
- ✓ 2026-09: production diffusion pipeline, RTX 5090: corrected decoder on the same latent mean 0.078/255, max 3/255.
