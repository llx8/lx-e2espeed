---
tier: on-demand
read_when: profiling shows the symptom below
---
# GPU preprocessing

- Category: data
- Applies to: image pipelines whose masks, crops, warps, and normalization run on the CPU before a GPU model

## Symptom
The time breakdown shows CPU preprocessing (mask cleanup, connected components, crop, warp, normalize) larger than the model itself; the GPU idles while the CPU works.

## Technique
Upload the input once and run the preprocessing chain on the GPU: PyTorch or Kornia operators first, Triton kernels where the exact semantics (for example connected-component cleanup or a warp) have no mature operator. Keep the result on the GPU and hand it to the engine directly. See the `gpu-inference-dataflow` skill.

## Effect
Preprocessing-heavy paths: 5x to 7x on the preprocessing share; lossless when the kernels replicate the reference exactly.

## Side effects
Triton kernels need a compiler at first run and a cold cache costs about 90 s. Triton also compiles a new variant per specialization: a block size derived from the image width compiled at the first very wide request (39 to 46 s, over a 10 s worker timeout), and a view whose pointer was not 16-byte aligned cost 2 s once. Use a fixed block with a loop over columns (cold Init 57.5 -> 9.9 s), clone unaligned views, and warm up every variant at Init. Bit-exactness can tie the code to a library version and CPU features of the reference (for example an OpenCV version with AVX2/FMA3).

## Does not apply
- A cheap CPU stage: moving it only to remove a visible transfer does not pay.

## Evidence
- ✓ 2026-09: image-to-guidance-map CNN service, B1 RoI 1024x1024, RTX 5090, TensorRT 11.3, full-resolution 4480x6720 input: one RoI 110.6 -> 19.5 ms end to end, L0 (byte-identical in 9 scenarios).
- ✓ 2026-09: two-network image color-transfer model at 1536x1536 with D=16 attention, RTX 5090, TensorRT 10.16: preprocessing 134 -> 19 ms with GPU (Kornia) preprocessing, within the model's accepted difference.
- ✓ 2026-09, pilot rerun of the same model: only the full-resolution normalize and uint8 conversion moved to the GPU, same float32 operations: -24 ms (-2.6%), L0.
