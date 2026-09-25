---
tier: on-demand
read_when: profiling shows the symptom below
---
# Pack the model input once

- Category: data
- Applies to: CPU preprocessing that builds a multi-channel float input from uint8 images and masks with cvtColor, astype, transpose, and concatenate

## Symptom
A 5x1024x1024 float32 input takes 10 to 20 ms of CPU time: each channel is converted, normalized, transposed, and copied several times, plus a contiguous copy before the engine.

## Technique
Allocate the NCHW float32 input once and write each channel with one ufunc call and `out=` (`np.divide(src[:, :, 2 - c], np.float32(127.5), out=x[0, c])`), keeping the original float32 operation order so the result is bit-identical; select channels by index instead of `cvtColor`.

## Effect
About -18 ms per RoI at 1024x1024, -35% end to end in one service; L0.

## Side effects
None; the engine input stays FP32.

## Does not apply
- Writing into the runner's existing pinned host buffer instead of a fresh pageable array: slower in one project (see `transfer/pinned-staging.md`).

## Evidence
- 2026-09, pilot rerun on image-to-guidance-map CNN service, B1 RoI 1024x1024, RTX 5090, TensorRT 11.3, 2048 input: two_instances 50.4 -> 32.7 ms (-35.2%), one_instance -40.8%, L0.
