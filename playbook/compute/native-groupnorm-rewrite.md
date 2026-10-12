---
tier: on-demand
read_when: profiling shows the symptom below
---
# Native GroupNorm from the InstanceNorm export pattern

- Category: compute
- Applies to: ONNX exports where GroupNorm became Reshape -> InstanceNormalization -> Reshape (PyTorch exporters do this), built with TensorRT 10 on sm_120; no MPS deadlock in play

## Symptom
TensorRT runs the normalizations as Myelin InstanceNorm kernels with reshapes around them, and the affine and activation that follow do not fuse into them.

## Technique
Rewrite only the exact exporter pattern: a Reshape whose shape initializer is [0, G, -1] feeding an InstanceNormalization with scale all 1 and bias all 0, consumed once. Drop that Reshape, replace the InstanceNormalization with ONNX GroupNormalization(num_groups=G, epsilon as exported) on the 4-D input, and keep the second Reshape and every downstream affine and SiLU so TensorRT fuses them. Assert the replacement count against the network's GroupNorm count. A few dozen lines of onnx.helper; the project that did it keeps the tool.

## Effect
One network of an image model: 422 -> 357 ms (-15%), FP16 engine; outputs against the production FP16 engine MAE 0.08 levels, max 4, PSNR 58.9 dB.

## Side effects
The output changes within FP16 noise, so the user accepts it like any precision change. Needs opset 18 or later for GroupNormalization. The native op still lowers to Myelin kernels with a grid barrier: under MPS, `tensorrt-groupnorm-plugin` instead.

## Does not apply
- Other graph shapes for the same op (the reshape kept around GroupNormalization, a direct 4-D rewrite that was not the strict pattern): FP16 467 and 400 ms, BF16 516 and 686 ms, PSNR 7.5 to 47 dB.
- Forcing the InstanceNorm to float in the original graph: 422 ms either way.
- FP32 reductions around the native GroupNorm (Cast in and out): 358 ms and max error 5, worse than plain.
- A single-block fused GroupNorm written for the model: 498 ms, PSNR 20.6 dB.
- cuDNN InstanceNorm tactics: refused by TensorRT on Blackwell.

## Evidence
- ✓ 2026-09: two-network image color-transfer model at 1536x1536, RTX 5090, TensorRT 10, FP16: 422 -> 357 ms on one network; against the production FP16 engine MAE 0.0825, max 4, PSNR 58.95 dB.
