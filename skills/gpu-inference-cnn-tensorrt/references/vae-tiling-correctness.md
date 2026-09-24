# TensorRT VAE Tiling Correctness

## Preserve the algorithm, not only the output shape

Diffusers VAE tiling is part of the model's effective algorithm. Freeze and record all of the following before conversion:

- `tile_sample_min_size`, `tile_latent_min_size`, compression ratio, and overlap factor;
- tile traversal starts and the exact short shapes at the right and bottom edges;
- encoder and decoder blend extent, crop limit, latent scaling, and posterior selection;
- dtype, attention precision, normalization, input range, and output clamp/rounding.

Do not assume a remembered Diffusers default. Read the values from the loaded VAE. A model configured for a 1024 sample tile is not semantically equivalent to the same weights run with a 512 tile. The smaller tile changes the effective context throughout the image, not only at seams.

## Static B1 does not require static H/W

Keep the service batch dimension fixed at 1. Choose the spatial contract independently:

- Use static H/W only when every runtime tile has that shape or padding equivalence is proven on the real VAE.
- Use a bounded dynamic H/W profile when Diffusers emits true short edge tiles. Set encoder minimum H/W to the VAE spatial scale factor, decoder minimum latent H/W to 1, and opt/max to the reference tile sizes.
- Execute the original Diffusers traversal and blend code around the TensorRT encoder/decoder. Do not replace it with a different full-tile weighting scheme merely because final dimensions match.

Zero-padding a short tile to the static maximum and cropping the output is generally invalid for a deep VAE. Convolutions and normalization process the artificial region, and the contamination propagates back into valid pixels. Reflect padding is also a different boundary condition and requires evidence rather than intuition.

## Dynamic GroupNorm

Dynamic spatial profiles can make expanded TensorRT GroupNorm graphs slow and memory-heavy. Use the maintained SeedVR/Klein GroupNorm or GroupNorm+SiLU rewrite and plugin when the exported graph matches its contract. Require a nonzero rewrite count and include the plugin identity in the engine cache key.

If decoder mid-block attention must be upcast to FP32 for finite output, do not route its FP32 GroupNorm through an FP16-only plugin. Exclude that named attention GroupNorm while rewriting the FP16 convolutional body. Validate the exclusion against the exported node names and fail the build if the expected rewrite/exclusion pattern changes.

## Validation ladder

1. Compare one full encoder tile between eager and TensorRT.
2. Compare one full decoder tile using the same latent.
3. Compare true short edge tiles independently in both axes and at the bottom-right corner.
4. Compare tiled encoder moments and selected latent.
5. Decode the same eager latent through eager and TensorRT to isolate decoder error.
6. Compare the complete TensorRT encode/decode path.
7. Run the complete downstream diffusion pipeline with the same input and seed.

Report image-space mean/max error as well as latent/tensor error. A VAE reconstruction that looks close in isolation can still alter conditioning enough for a later diffusion stage to produce a visibly different image.

## Workspace and OOM diagnosis

Dynamic max shapes can increase TensorRT context device memory. Inspect `device_memory_size_v2` for encoder and decoder plans and confirm every non-concurrent context uses the same shared pool. The pool size is the maximum live requirement, not the sum, only when contexts truly cannot overlap.

Before attributing an OOM to a larger shared workspace, reconcile all three views:

- total GPU usage from `nvidia-smi`;
- per-process GPU usage and command line;
- the target process's PyTorch allocated/reserved values plus TensorRT non-PyTorch memory.

If total usage is much larger than the target process, identify external GPU processes first. Do not redesign model lifetimes or reduce quality based on an OOM caused by an unrelated process.

## Case evidence

In the production image pipeline regression that motivated these rules, forcing a Flux1 VAE from its loaded 1024/128 tile geometry to 512/64 and padding short tiles produced the following real-image reconstruction errors against Diffusers eager:

- broken 512 static path, complete encode/decode: mean `5.02/255`;
- corrected B1 dynamic-H/W 1024 path with native edge shapes and GroupNorm plugin: mean `0.138/255`;
- corrected decoder using the same eager latent: mean `0.078/255`, max `3/255`.

The apparent deployment OOM was initially blamed on the corrected decoder's `3.55 GiB` context requirement. Process inspection showed an unrelated seed-sweep process occupying `16.56 GiB`; after it exited, the original two-Stage3 model layout completed successfully with one shared TensorRT workspace.
