---
tier: on-demand
read_when: profiling shows the symptom below
---
# GPU JPEG decode

- Category: data
- Applies to: services that receive JPEG bytes and process the image on the GPU

## Symptom
Service-side JPEG decode dominates the request (over 75% of the service chain), followed by an upload of the decoded image.

## Technique
Decode with nvImageCodec (NVJPG hardware) straight into a resident CUDA tensor, keep one codec instance, and let the algorithm accept that tensor directly. Encode from the CUDA tensor and synchronize only before reading the bytes. See the `gpu-inference-dataflow` skill.

## Effect
Service end to end 8 to 16 ms faster per request.

## Side effects
The algorithm's input type widens (a CUDA tensor besides a NumPy array); the service and algorithm repos must change together.

## Does not apply
- Paths that need the decoded image on the CPU anyway.

## Evidence
- ✓ 2026-09: image-to-guidance-map CNN service, B1 RoI 1024x1024, RTX 5090, TensorRT 11.3, 2k service requests: end to end 8.3 to 16.0 ms faster, responses byte-identical.
