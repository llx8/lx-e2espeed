---
tier: on-demand
read_when: profiling shows the symptom below
---
# Warm the whole request path at Init

- Category: host
- Applies to: GPU services whose first request pays one-time costs: a hardware decoder's first use, first engine and kernel runs, lazy imports, JIT compiles

## Symptom
The first request after start takes 1 to 2 s against tens of ms warm. Split by stage in one service (1.5 s): the first NVJPG hardware JPEG decode (about 0.9 s against 1.7 ms warm), the first engine and solver runs, and about 0.2 s importing the compiler stack, which torch.compile wrappers do on their first call even when a stance runs them eagerly. Its Init warm-up read a file the repo never had and only logged a warning.

## Technique
In Init, send one real input per decode path (a JPEG for the hardware decoder, a PNG for the other) through the whole request path, preprocessing, inference, and postprocessing, inside the same compile stance the requests use. Fail Init on any error or empty result instead of logging it. Then time the first request by stage and check it costs about a warm one.

## Effect
First request 1546 -> 29.8 ms (RTX 5090) and 1523-1565 -> 25.5 ms (L20), a warm request plus 1 to 2 ms; warm latency and outputs unchanged (L0).

## Side effects
Init about 1.7 s longer: the costs move there. Each later new input shape still pays on first use (+1.7 ms here without CUDA graphs, +37 ms with a graph captured per shape). JIT-compiled kernels (Triton) compile here too: +1.6 to 2.1 s with an empty cache.

## Does not apply
- None measured yet.

## Evidence
- ✓ 2026-10: camera-calibration CNN-transformer service with an iterative least-squares refinement, batch 1, hardware JPEG decode, RTX 5090 and L20: first request 1546 -> 29.8 ms and 1523-1565 -> 25.5 ms, Init +1.7 s, L0.
