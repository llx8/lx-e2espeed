---
tier: on-demand
read_when: profiling shows the symptom below
---
# NumPy temporaries

- Category: data
- Applies to: CPU code that counts, casts, or compares full-resolution arrays

## Symptom
Simple statistics on a large mask take milliseconds: `np.sum(mask == 255)` builds a full boolean array, and `mask.astype(np.uint8)` copies an array that is already uint8.

## Technique
Use a primitive that does not materialize a temporary (`cv2.countNonZero` when the mask is 0/255), and `astype(..., copy=False)`. Profile per instance: such calls often run once per object.

## Effect
One count 12.7 -> 0.94 ms; one avoided copy 7.7 ms. Together about 18% of an end-to-end request in one case.

## Side effects
None when the value domain is confirmed (0/255 masks).

## Does not apply
- Paths already moved to the GPU.

## Evidence
- 2026-09: image-to-guidance-map CNN service, B1 RoI 1024x1024, RTX 5090, TensorRT 11.3, 4480x6720 masks, per instance: count 12.7 -> 0.94 ms, uint8 copy of 28.7 MiB 7.7 ms -> 0.
