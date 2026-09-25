---
tier: on-demand
read_when: profiling shows the symptom below
---
# Integer mask cleanup

- Category: data
- Applies to: CPU mask cleanup that thresholds uint8 masks through float division, or removes small connected components with a per-component loop

## Symptom
Mask cleanup costs milliseconds per full-size mask: `mask / 255 < t` builds float arrays, and `out[labels == i] = 255` compares the whole image once per component.

## Technique
For uint8, turn `x / 255 < t` into the equivalent integer threshold and check it for all 256 values; drop steps a later threshold dominates. For small-component removal, keep the thresholded foreground and zero only the small components inside their own bounding boxes (`stats` from `connectedComponentsWithStats`). Test the zeroing branch with masks that have small components: real masks may have none.

## Effect
Cleanup stage about 17 -> 5 ms at a 2048 input; L0.

## Side effects
None; output identical.

## Does not apply
- A lookup table over the label image (`keep[labels]`): a full int32 gather that cost about 7 ms, more than zeroing a few boxes.

## Evidence
- 2026-09, pilot rerun on image-to-guidance-map CNN service, B1 RoI 1024x1024, RTX 5090, TensorRT 11.3, 2048 input: integer thresholds -5.5 ms (-9.7%), then box-local zeroing -6.7 ms (-20.6%), L0 in all scenarios and in a branch test with small components.
