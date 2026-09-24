---
tier: on-demand
read_when: profiling shows the symptom below
---
# Upload only the region of interest

- Category: transfer
- Applies to: pipelines that crop regions from a large image on the GPU

## Symptom
The whole full-resolution image is uploaded per request although the model reads only a few regions.

## Technique
Compute the union of the regions first, upload only that slice, and offset the coordinates. When the image already arrives as a CUDA tensor, take a view instead.

## Effect
About 4 ms per request on a 30 MB image; lossless.

## Side effects
Coordinates must be offset consistently; test regions at the image border.

## Does not apply
- Regions covering most of the image.

## Evidence
- 2026-09: image-to-guidance-map CNN service, B1 RoI 1024x1024, RTX 5090, TensorRT 11.3, 4480x6720 input, one RoI: about 4.2 ms saved per new image, L0.
