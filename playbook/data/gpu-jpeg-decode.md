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
Decode with nvImageCodec (NVJPG hardware) straight into a resident CUDA tensor, keep one codec instance, and let the algorithm accept that tensor directly. Give the container the video driver capability (`NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`): without it the decoder cannot be created. Create the decoder at Init and fail there when it cannot be. Encode from the CUDA tensor and synchronize only before reading the bytes.
With several workers use nvImageCodec 0.9 or newer: 0.8 ran single-image hardware decodes about one at a time (about 890 2.8 MP images/s on a GPU with four NVJPG engines), 0.9 spreads them (about 2200/s), same pixels.

## Effect
Service end to end 5.8 to 16 ms faster per request (-33% to -83%); with GPU preprocessing after it, 11.35 -> 1.93 ms in a small-crop service.

## Side effects
The algorithm's input type widens (a CUDA tensor besides a NumPy array); the service and algorithm repos change together. NVJPG decodes up to ±1 level differently from libjpeg (L1 in one project, accepted as not L0 in another). Inputs NVJPG does not take (other formats, progressive, truncated) fail fast with an error response by default. Where the user wants them served, route them by file header to the original CPU decoder and record the reason per request (three services chose this).
A bit-exact path (the hardware planes plus a kernel with libjpeg-turbo's upsampling and color conversion) was built in four services and taken back in all four for the hardware decoder's ±1 levels: equal speed, more code.

## Does not apply
- Paths that need the decoded image on the CPU anyway.
- Throughput past one NVJPG engine on a GPU that has one: four JPEG services on an RTX 5090 stopped at about 980 2.8 MP images/s from 4 workers, whatever the SM load.

## Evidence
- ✓ 2026-09: image-to-guidance-map CNN service, B1 RoI 1024x1024, RTX 5090, TensorRT 11.3, 2k service requests: end to end 8.3 to 16.0 ms faster, responses byte-identical.
- ✓ 2026-10: DETR-style pose service, 2048-px JPEGs, RTX 5090, with the letterbox on the GPU: 16.44 -> 10.54 ms (-35.9%), not L0 from the decode, accepted.
- ✓ 2026-10: face classification CNN service, real requests with 1 to 28 faces, RTX 5090, with one fused preprocessing kernel: 11.35 -> 1.93 ms, L1.
- ✓ 2026-10: image background-type CNN classifier service, 2.8 MP JPEGs, RTX 5090, resize on the GPU after it: -7.51 ms (-83.1%), values change through the decode (max 7.6e-5), accepted.
- ✓ 2026-10: heatmap line-segmentation CNN service, the same inputs, RTX 5090, resize on the GPU after it: -6.89 ms (-65.9%), 3 of 100 responses moved, accepted.
- ✓ 2026-10: ViT image feature encoder service, the same inputs, FP16 engine, RTX 5090: -8.66 ms (-82.9%), lowest cosine 0.99999956.
- ✓ 2026-10: two-model mask service (segmentation transformer and classifier), JPEG image plus PNG masks, RTX 5090: -5.85 ms (-32.7%), L0 on its requests.
