---
tier: on-demand
read_when: profiling shows the symptom below
---
# GPU PNG decode

- Category: data
- Applies to: services that receive PNG masks or images, for example 8-bit grayscale masks written by `cv2.imencode`; a CUDA GPU

## Symptom
`cv2.imdecode` of a PNG is the largest preprocessing item. Most of it is libpng's byte-by-byte filter reversal, not zlib: 5.5 ms for one mask, 1.5 ms of it zlib.

## Technique
Decode on the GPU with a PNG decoder built on nvCOMP (deflate on the GPU), straight into a device tensor that goes to the model without a copy back. Build the decoder as a wheel in the build image, ship it with the service, and check it at Init with one image per filter type (Sub, Paeth). Inputs it does not take (multi-channel, corrupt, over its size limit) fail fast, unless the user agrees on another route.

## Effect
-4.8 ms on a real request (8.96 -> 4.13 ms end to end); by the image's long side, 512 to 6144 px: -0.4 to -78 ms (93.4 -> 15.8 ms at 6144). Responses byte-identical.

## Side effects
A native dependency (the nvCOMP wheel, the decoder built per CUDA major). Above 3072 px the JPEG decode becomes the bottleneck.

## Does not apply
- CPU zlib plus GPU unfiltering with a hand-written PNG parser (Sub as `cumsum(dim=1, dtype=torch.uint8)`, Up along dim 0): -3.6 ms, but reverted at the user's request; zlib stays on the CPU, since no faster zlib (isal, zlib-ng) was installable.
- The JPEG hardware decoder: it does not decode PNG.

## Evidence
- ✓ 2026-09: image service with PNG fallback masks, RTX 5090, CUDA 13, TensorRT 10.16: 8.96 -> 4.13 ms, 59 requests byte-identical.
- ✓ 2026-10: image background-type CNN classifier service, 8-bit PNG requests, RTX 5090: -3.72 ms (-58.8%) on the PNG scenario, L0.
- ✓ 2026-10: two-model mask service, PNG masks in each request, RTX 5090: -5.18 ms (-19.8%), L0.
