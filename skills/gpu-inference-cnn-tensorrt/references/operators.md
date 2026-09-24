# Mature Operator and Graph Mechanisms

Keep only operators that a target project actually needs and can benchmark.

## GroupNorm and activation

For restoration networks where GroupNorm or GroupNorm+SiLU blocks prevent a good engine, use a maintained ONNX graph replacement and TensorRT plugin with explicit channel, epsilon, dtype, and layout contracts. Test non-divisible spatial boundaries used by tiles.

## Attention inside CNN/restoration

Use TensorRT-native layers first. Introduce an attention plugin only for a measured unsupported or dominant pattern with fixed dimensional contracts. Validate softmax stability and accumulation precision.

## Resize and sampling

Use TensorRT or a mature GPU library when its coordinate transformation, antialiasing, and rounding match the reference. Do not replace image resize based only on a microbenchmark.

## Masked restoration

Crop or tile the effective masked region, skip empty tiles, run the static engine, and blend directly on GPU. Avoid allocating a full-resolution temporary per instance or tile.

## Convolutional VAE

Keep VAE batch fixed at B1. Preserve the model's reference tile size, latent scaling, attention precision, edge-tile shapes, traversal, and blend behavior. Use bounded dynamic H/W profiles when static edge padding is not numerically equivalent. See [vae-tiling-correctness.md](vae-tiling-correctness.md).

## Deformable convolution segmentation

For BiRefNet-style networks that use `torchvision.ops.deform_conv2d`, register the pinned `deform_conv2d_onnx_exporter==1.2.0` symbolic before exporting `torchvision.ops.deform_conv2d`. PyTorch 2.9 exposes HALF as a `JitScalarType` enum rather than the legacy string key expected by that exporter; explicitly map it to ONNX `FLOAT16`. Treat the resulting engine as static B1/1024 unless a separately validated profile is required.
