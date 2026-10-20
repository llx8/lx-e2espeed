---
tier: on-demand
read_when: profiling shows the symptom below
---
# ONNX export with the torch.export-based exporter

- Category: env
- Applies to: PyTorch 2.11 or later exporting ONNX for TensorRT, where the TorchScript-based exporter (`dynamo=False`) is deprecated

## Symptom
The export warns that the TorchScript-based exporter is deprecated, or the owner wants one exporter everywhere; the new exporter fails on code the old one traced.

## Technique
Export with `dynamo=True` (opset 20, batch and spatial dims as `Dim.DYNAMIC`, `external_data=False` for one file) and pin onnxscript to the tested version in the build script. Fix what it rejects at the source: shape reads such as `torch._shape_as_tensor` (branch on tracing, or use the size) and `torch.where` with Python scalar branches (pass float tensors).
Normalizations come out as native LayerNormalization nodes: re-match any per-layer precision pins and assert their count from the built engine. Measure per GPU: engine execution memory (`device_memory_size_v2`), engine-only and end-to-end speed, export and build time.

## Effect
Speed equal within noise for a CNN-transformer network (its engine alone -0.17 ms on the L20); outputs L1 (another graph), 100 images within 0.1 px P50.

## Side effects
Network engine execution memory +390.6 MiB per process on the L20, every build, at every shape; none on the RTX 5090. Export 14 -> 36 s, the CI build step 379 -> 352 s. A large unrolled solver graph took 181 s to export against 8 s.

## Does not apply
- An unrolled iterative-solver graph: exported and built equal (L0), but its engine was 0.7 to 0.8 ms slower on the L20 at the main shape.

## Evidence
- ✓ 2026-10: camera-calibration CNN-transformer service, FP16 network engine, TensorRT 10.16, L20 and RTX 5090: switched; L20 A/B +0.01 and -0.15 ms (inside noise), L1, +390.6 MiB on the L20.
- ✗ 2026-10: the same service's unrolled solver engine, FP32: L20 +0.20 ms end to end (noise 0.06), engine +0.7 to +0.8 ms at 480x320; dropped (the solver moved to Triton kernels).
