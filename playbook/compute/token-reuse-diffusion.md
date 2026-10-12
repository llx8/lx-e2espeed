---
tier: on-demand
read_when: profiling shows the symptom below
---
# Token reduction with cross-step reuse in diffusion transformers

- Category: compute
- Applies to: diffusion transformers at high resolution; it changes outputs, so only as a research trial where PROJECT.md allows one (`rules/self-iteration.md`)

## Symptom
Attention and MLPs over many image tokens dominate every denoising step, and consecutive steps produce similar branch outputs.

## Technique
In the DiTo style: at a matching step, run the branch in full and map each source token to its most similar destination (cosine of the branch outputs, one destination per 2x2 cell); at reduction steps, compress the input by that map before the QKV projection, run the short sequence, and copy the outputs back. A schedule calibrated offline on dense trajectories (overlap of top-k matches across steps) decides when to match again. Text tokens stay full.

## Effect
At the paper's image quality, 2% to 5% end to end (the paper claims 9%).

## Side effects
Outputs change everywhere, PSNR 22 to 32 dB against dense depending on the schedule; the failure is drifting composition, not local artifacts. Calibrate per model and step count.

## Does not apply
- Reducing from the first step: early matches do not hold (overlap 0.46 to 0.84 a few steps later) and the composition drifts (18.7 dB). Keep the first 5 steps dense.
- Few-step models: with 8 steps only 3 or 4 reduce steps stayed acceptable, 3% to 4%.
- Without fused kernels: a plain PyTorch attention processor was 4% slower than the library's fused one before any reduction, and MLPs and output projections left at full length keep most of the time. Measure matching, gather, and recovery in the real time, never estimate from N².
- Penalty strength, history shared across layers, prune or mean merging: within 0.5 dB of each other; not the lever.

## Evidence
- ✗ 2026-09: paper reproduction, FLUX.1-dev bf16, 1024x1024, 35 steps, ratio 0.25, SDPA, RTX 5090: 22.2 -> 21.5 s at 26.8 dB against dense (5 prompts, one seed).
