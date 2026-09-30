---
tier: on-demand
read_when: looking for optimization ideas after profiling
---
# Playbook index

Find the category where profiling shows the time goes, then open only the matching cards.
Each `## name` heading below is a category and the name of its directory.
Entry format, a few words each: `- [<technique>](<category>/<technique>.md): <symptom> → <effect> | ✓N ✗M`, where N and M count the card's Evidence lines marked ✓ (worked) and ✗ (failed); the checker recounts them.

## data
Loading, decoding, tokenization, CPU-side preprocessing.
- [gpu-preprocessing](data/gpu-preprocessing.md): CPU preprocessing dominates → 5-7x, L0 | ✓3 ✗0
- [exact-resize-on-gpu](data/exact-resize-on-gpu.md): CPU mask resize → 5x, L0 | ✓1 ✗0
- [gpu-jpeg-decode](data/gpu-jpeg-decode.md): JPEG decode dominates → -8 to -16 ms | ✓1 ✗0
- [numpy-temporaries](data/numpy-temporaries.md): full-array temporaries, copies → up to -50 ms | ✓2 ✗0
- [mask-cleanup-integer](data/mask-cleanup-integer.md): float thresholds, per-label loops → cleanup 3x, L0 | ✓1 ✗0
- [pack-model-input](data/pack-model-input.md): input built by cvtColor/transpose/concat → -35%, L0 | ✓1 ✗0

## host
Python overhead, kernel launch overhead, unnecessary synchronization, CUDA Graphs.
- [fused-preprocess-kernel](host/fused-preprocess-kernel.md): many small prep kernels → -0.3 ms/RoI, L0 | ✓1 ✗0
- [cuda-graph-trt](host/cuda-graph-trt.md): enqueue overhead → -0.1 to -0.4 ms, L0 | ✓1 ✗0

## transfer
Host-device copies, pinned memory, asynchronous copies, overlap.
- [roi-only-upload](transfer/roi-only-upload.md): whole image uploaded → -4 ms, L0 | ✓1 ✗0
- [pinned-staging](transfer/pinned-staging.md): pageable uploads → 2.8x copy; ask first | ✓1 ✗2
- [batched-copy-back](transfer/batched-copy-back.md): per-RoI sync → -0.5 ms, L0 | ✓1 ✗0

## compute
Operator fusion, torch.compile, kernel selection, custom kernels.
- [fp16-trt-engine](compute/fp16-trt-engine.md): FP32 engine dominates → 2.2x, L2; FP32 edges eat it | ✓1 ✗1
- [tensorrt-groupnorm-plugin](compute/tensorrt-groupnorm-plugin.md): Myelin norm kernels, MPS hang → fixed, up to 1.8x | ✓1 ✗0
- [quantized-attention-plugin](compute/quantized-attention-plugin.md): long D=16 attention on sm_120 → 2.9x, max 5 levels | ✓1 ✗0
- [torch-compile](compute/torch-compile.md): eager stages, no mature backend → ~1% | ✓0 ✗1

## memory
Allocator behavior, fragmentation, buffer reuse.
- [vae-tiling-trt](memory/vae-tiling-trt.md): TensorRT VAE drifts or OOMs → 5.02 to 0.138/255 | ✓1 ✗0

## serving
Batching, concurrency, scheduling, pipelining.
- [mps-grid-barrier](serving/mps-grid-barrier.md): MPS workers hang → fixed | ✓1 ✗0

## env
Driver and library versions, build flags, GPU settings.
- [trt-plugin-build-hygiene](env/trt-plugin-build-hygiene.md): plugin not found, binaries overwritten → prevented | ✓1 ✗0
