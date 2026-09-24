---
tier: on-demand
read_when: profiling shows the symptom below
---
# torch.compile on large eager models

- Category: compute
- Applies to: eager PyTorch models where a mature backend is not an option; only with explicit approval

## Symptom
Eager model stages dominate and TensorRT or another mature backend is not available for them.

## Technique
Compile the smallest repeated hotspot with a fixed shape policy, measured in fresh processes. Follow the approval gate in the `gpu-inference-torch-compile` skill.

## Effect
One case: -1.3% end to end, within the stage noise of that time.

## Side effects
First compilation took about 100 s; cold start, cache identity, and maintenance.

## Does not apply
- When a TensorRT path exists: it gave 2.9x on the same model.
- When cold start matters or shapes vary per request.

## Evidence
- 2026-08: two-network image color-transfer model at 1536x1536 with D=16 attention, RTX 5090, TensorRT 10.16: 2316 -> 2285 ms, gain swallowed by variance in another stage; not adopted.
