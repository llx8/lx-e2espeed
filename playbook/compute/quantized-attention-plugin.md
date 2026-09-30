---
tier: on-demand
read_when: profiling shows the symptom below
---
# Quantized fused attention plugin

- Category: compute
- Applies to: high-resolution models dominated by attention with a small head dimension, on sm_120

## Symptom
Attention over very long sequences (1536x1536 feature maps) dominates the engine; stock kernels fit small head dimensions poorly.

## Technique
A custom TensorRT plugin for sm_120 in the SageAttention style: INT8 QK and FP8 PV with per-block quantization, fused kernels, then tuned for registers, synchronization, and TMA. Optional block-sparse routing skips low-weight tiles. Keep a dense baseline family and a candidate family, and apply every later change to both.

## Effect
With the TensorRT move: 1133 -> 390 ms end to end (2.9x). Block routing added 21% (tau -0.5, max error 5 levels) to 35% (tau 0, max 8).

## Side effects
Output differs: max 4 to 5 levels, PSNR 47 to 57 dB; the user must accept it. Large maintenance cost; the plugin binaries are tied to one GPU architecture.

## Does not apply
- K/V pooling: 40% to 61% faster but PSNR 35 to 41 dB; rejected by the user.
- FP4 QK in software, other Q/K group sizes, FP8 PV, and V layout variants: no gain or slower.
- A zero-spill build: slower than the tuned build.
- Switching the PyTorch SDPA backend instead: at D=16 on sm_120, efficient attention was 2.9x and cuDNN 1.9x slower than flash at every shape (pilot rerun). Measure this cheap option first; it settled quickly.

## Evidence
- ✓ 2026-09: two-network image color-transfer model at 1536x1536 with D=16 attention, RTX 5090, TensorRT 10.16: warm 1.006 -> 0.278 s, cold start 1.364 -> 0.653 s, peak memory 1199 -> 773 MiB.
