---
tier: on-demand
read_when: profiling shows the symptom below
---
# Fused deformable attention plugin

- Category: compute
- Applies to: multi-scale deformable attention (the Deformable-DETR family) in FP16 TensorRT engines

## Symptom
Around each deformable attention call, the engine runs the sampling-offset and attention-weight linears, the softmax, and the sampling locations as separate kernels before the sampling plugin.

## Technique
Replace the middle of the module, between value_proj and output_proj, with one plugin: both linears as one cuBLAS GEMM over FP16 weights concatenated offsets first, then softmax, locations, and sampling in one kernel. The `gpu-inference-trt-plugins` skill has the library, the export wrapper, and the checks.

## Effect
-0.43 ms (-4.5%) end to end on top of the IPluginV3 built-in; outputs within the FP16 rebuild spread.

## Side effects
Process memory +270 MiB (one cuBLAS handle per instance, 12 instances); cold start +0.19 s; libcublas at run time; the library's constraints (FP16, batch 1, 32 channels per head, 3 or 4 levels, 4 points).

## Does not apply
- The FP32 reference engine: the plugin is FP16 only, so that build keeps the built-in path.

## Evidence
- ✓ 2026-10: DETR-style pose service, 12 calls, RTX 5090, TensorRT 10.16: 9.48 -> 9.05 ms, L2 as every FP16 rebuild, no grid-barrier kernel.
