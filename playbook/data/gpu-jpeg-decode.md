---
tier: on-demand
read_when: profiling shows the symptom below
---
# GPU JPEG decode

- Category: data
- Applies to: services that receive JPEG bytes and process the image on the GPU

## Symptom
Service-side JPEG decode dominates the request (over 75% of the service chain in one service, 6.6 of 16.4 ms in another), followed by an upload of the decoded image.

## Technique
Decode with nvImageCodec (NVJPG hardware) straight into a resident CUDA tensor, keep one codec instance, and let the algorithm accept that tensor directly. Give the container the video driver capability (`NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`): without it the decoder cannot be created. Create the decoder at Init and fail there when it cannot be. Encode from the CUDA tensor and synchronize only before reading the bytes. See the `gpu-inference-dataflow` skill.

## Effect
Service end to end 5.9 to 16 ms faster per request; with GPU preprocessing after it, 11.35 -> 1.93 ms in a small-crop service.

## Side effects
The algorithm's input type widens (a CUDA tensor besides a NumPy array); the service and algorithm repos change together. NVJPG decodes up to ±1 level differently from libjpeg (L1 in one project, accepted as not L0 in another). Inputs NVJPG does not take (other formats, progressive, truncated) fail fast with an error response, the user's default; never the original decoder.

## Does not apply
- Paths that need the decoded image on the CPU anyway.

## Evidence
- ✓ 2026-09: image-to-guidance-map CNN service, B1 RoI 1024x1024, RTX 5090, TensorRT 11.3, 2k service requests: end to end 8.3 to 16.0 ms faster, responses byte-identical.
- ✓ 2026-10: DETR-style pose service, 2048-px JPEGs, RTX 5090, with the letterbox on the GPU: 16.44 -> 10.54 ms (-35.9%), not L0 from the decode, accepted.
- ✓ 2026-10: face classification CNN service, real requests with 1 to 28 faces, RTX 5090, with one fused preprocessing kernel: 11.35 -> 1.93 ms, L1.
