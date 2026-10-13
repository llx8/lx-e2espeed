---
tier: on-demand
read_when: profiling shows the symptom below
---
# GroupNorm as a TensorRT plugin

- Category: compute
- Applies to: networks with GroupNorm or InstanceNorm exported to TensorRT (restoration CNNs, VAEs, segmentation heads, detection-transformer input projections)

## Symptom
TensorRT turns ONNX GroupNormalization or InstanceNormalization into slow Myelin kernels (`__myl_Gnor*`, `__myl_Inor*`, `*MeanMean*`); under multi-process MPS the service can deadlock.

## Technique
Replace the normalization nodes with a plugin that reduces within a block, with explicit channel, epsilon, dtype, and layout contracts; include the plugin identity in the engine cache key. First look for a tested compiled library (this machine's are listed in `lx-e2espeed/local/site.md`); [mps-grid-barrier](serving/mps-grid-barrier.md) explains the deadlock.

## Effect
Removes the MPS deadlock; latency from unchanged (+0.07 ms, inside noise) to one embedder 9.4 -> 5.1 ms with no FP32 lock needed for precision.

## Side effects
A plugin library ships with the service and must match the TensorRT ABI and the compute capabilities. The plugin splits the Myelin graph where it sits.

## Does not apply
- Rewriting GroupNorm as InstanceNormalization: same barrier kernels.
- Native GroupNorm variants with FP32 reductions, or a naive single-block fused kernel: slower or failed quality in one project.

## Evidence
- ✓ 2026-09: several production engines had 4 to 14 barrier kernels each; after the plugin, 6 workers with MPS at concurrency 12 ran 8 stress rounds without a hang, latency unchanged, worst 0.055% of pixels off by more than one level.
- ✓ 2026-10: DETR-style pose model, 4 GroupNorms, a colleague's compiled FP16 plugin, RTX 5090, TensorRT 10.16: barrier kernels 1 -> 0, 9.43 -> 9.50 ms (inside noise), outputs within the FP16 rebuild spread.
