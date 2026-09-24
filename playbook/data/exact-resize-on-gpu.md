---
tier: on-demand
read_when: profiling shows the symptom below
---
# Exact resize on the GPU

- Category: data
- Applies to: masks or images resized with OpenCV or PIL on the CPU where the output must not change

## Symptom
A CPU `cv2.resize` or PIL resize of a full-size mask sits on the hot path, followed by an upload of the large result.

## Technique
Upload the small input and resize on the GPU with a kernel that replicates the CPU library's coordinate mapping, weights, and rounding bit for bit (Triton). Check the replica against the CPU output on the real sizes. The `gpu-inference-dataflow` skill maps CPU interpolation modes to PyTorch ones.

## Effect
Resize stage 5x; lossless.

## Side effects
A replica is maintenance: it must be re-verified when the CPU library version changes.

## Does not apply
- Swapping in `F.interpolate` bilinear when the output must stay the same: the final output differed by up to 51 levels and the demo ran 5x slower.
- Test inputs whose mask already has the image size: the CPU resize returns early there, so they never exercise the path production uses.

## Evidence
- 2026-09: image-to-guidance-map CNN service, B1 RoI 1024x1024, RTX 5090, TensorRT 11.3, mask at half the image size: resize stage 11.6 -> 2.3 ms; stage with upload 28.8 -> 15.4 ms; L0 across 19 scenarios and a real request.
