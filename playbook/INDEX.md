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
- [gpu-preprocessing](data/gpu-preprocessing.md): CPU pre- or postprocessing dominates → 5-7x, L0 | ✓4 ✗0
- [exact-resize-on-gpu](data/exact-resize-on-gpu.md): CPU mask resize → 5x, L0 | ✓1 ✗0
- [gpu-jpeg-decode](data/gpu-jpeg-decode.md): JPEG decode dominates → -6 to -16 ms; nvImageCodec 0.9 for workers | ✓7 ✗0
- [gpu-png-encode](data/gpu-png-encode.md): cv2 PNG encode of response masks → -2.2 then -1.6 ms, bytes L0 | ✓2 ✗0
- [decode-once-per-request](data/decode-once-per-request.md): same input decoded again in a request → -10 ms, L0 | ✓1 ✗0
- [gpu-png-decode](data/gpu-png-decode.md): PNG mask decode dominates → -3.7 to -5.2 ms, up to -78 ms, L0 | ✓3 ✗0
- [numpy-temporaries](data/numpy-temporaries.md): full-array temporaries, copies → up to -50 ms | ✓2 ✗0
- [mask-cleanup-integer](data/mask-cleanup-integer.md): float thresholds, per-label loops → cleanup 3x, L0 | ✓1 ✗0
- [pack-model-input](data/pack-model-input.md): input built by cvtColor/transpose/concat, full-size BGR convert → -15% to -35%, L0 | ✓3 ✗0

## host
Python overhead, kernel launch overhead, unnecessary synchronization, CUDA Graphs.
- [fused-preprocess-kernel](host/fused-preprocess-kernel.md): many small prep kernels → -0.3 ms/RoI, L0 | ✓1 ✗0
- [one-wait-per-request](host/one-wait-per-request.md): many small synchronizes per request → -1.2 to -1.6 ms, cores freed under load | ✓2 ✗2
- [engine-primary-context](host/engine-primary-context.md): engine in its own CUDA context → -0.25 ms, -428 MiB, L0 | ✓1 ✗0
- [overlap-cpu-stages](host/overlap-cpu-stages.md): independent CPU stages back to back → -0.5 to -5 ms, L0 | ✓2 ✗0
- [cuda-graph-trt](host/cuda-graph-trt.md): enqueue overhead → -0.07 to -1 ms, L0; not with per-request shapes | ✓9 ✗1
- [init-warm-up](host/init-warm-up.md): first request 1.5 s, warm 25 ms → first request a warm one + 2 ms, Init +1.7 s | ✓1 ✗0
- [blocking-sync-wait](host/blocking-sync-wait.md): one spinning core per worker under load → CPU 599% to 114% at 6 workers, req/s equal, +0.08 ms single | ✓1 ✗0

## transfer
Host-device copies, pinned memory, asynchronous copies, overlap.
- [roi-only-upload](transfer/roi-only-upload.md): whole image uploaded → -4 ms, L0 | ✓1 ✗0
- [pinned-staging](transfer/pinned-staging.md): pageable uploads → 2.8x copy; ask first | ✓1 ✗2
- [batched-copy-back](transfer/batched-copy-back.md): per-RoI sync or readback → -0.5 to -0.8 ms, L0 | ✓2 ✗0

## compute
Operator fusion, torch.compile, kernel selection, custom kernels.
- [fp16-trt-engine](compute/fp16-trt-engine.md): FP32 engine dominates → 2.2x to -62%, L2; FP32 edges eat it | ✓3 ✗2
- [tensorrt-groupnorm-plugin](compute/tensorrt-groupnorm-plugin.md): Myelin norm kernels, MPS hang → fixed, up to 1.8x | ✓2 ✗0
- [plugin-v3-version](compute/plugin-v3-version.md): reformats around a built-in plugin → -14.8% | ✓1 ✗0
- [deformable-attention-plugin](compute/deformable-attention-plugin.md): deformable attention glue kernels → -4.5%, +270 MiB | ✓1 ✗0
- [quantized-attention-plugin](compute/quantized-attention-plugin.md): long D=16 attention on sm_120 → 2.9x, max 5 levels | ✓1 ✗0
- [fused-attention-kernel](compute/fused-attention-kernel.md): custom attention kernel dominates → 1.45x to 2.86x per shape | ✓2 ✗0
- [token-reuse-diffusion](compute/token-reuse-diffusion.md): image tokens dominate DiT steps → 2-5% at PSNR 27 dB; research only | ✓0 ✗1
- [torch-compile](compute/torch-compile.md): eager stages, no mature backend → ~1% | ✓0 ✗1
- [engine-difference-triage](compute/engine-difference-triage.md): engine output off beyond the agreed level → one node or subgraph isolated | ✓2 ✗0
- [native-groupnorm-rewrite](compute/native-groupnorm-rewrite.md): InstanceNorm export pattern blocks fusion → -15%, FP16 noise | ✓1 ✗0
- [engine-graph-rewrite](compute/engine-graph-rewrite.md): one op form on a slow TensorRT kernel → L20 engine -0.14 to -0.18 ms; 5090 gains reversed on the L20 | ✓1 ✗2
- [iterative-solver-kernels](compute/iterative-solver-kernels.md): solver of thousands of tiny ops, GPU idle → solver 3.4 to 0.48 ms, request -32%, L1 | ✓4 ✗0
- [tiny-batched-gemm-elementwise](compute/tiny-batched-gemm-elementwise.md): cuBLAS batched GEMMs in per-pixel math → -21% to -48% on the L20, L0-L1 | ✓2 ✗0

## memory
Allocator behavior, fragmentation, buffer reuse.
- [vae-tiling-trt](memory/vae-tiling-trt.md): TensorRT VAE drifts or OOMs → 5.02 to 0.138/255 | ✓1 ✗0
- [memory-attribution](memory/memory-attribution.md): peak too high, per-stage deltas read zero → the stage that sets it | ✓1 ✗0

## serving
Batching, concurrency, scheduling, pipelining.
- [mps-grid-barrier](serving/mps-grid-barrier.md): MPS workers hang → fixed | ✓2 ✗0
- [mps-hang-forensics](serving/mps-hang-forensics.md): workers hung under MPS, no debugger attaches → root cause from one GPU coredump | ✓1 ✗0
- [mps-multi-worker](serving/mps-multi-worker.md): throughput flat as workers grow → 3 workers 2.6-3.1x, 12 workers 3.5-7.2x, L0 | ✓7 ✗0
- [dynamic-batch-profiles](serving/dynamic-batch-profiles.md): one engine call per item → P99 -39%, L1 | ✓1 ✗0

## env
Driver and library versions, build flags, GPU settings.
- [trt-plugin-build-hygiene](env/trt-plugin-build-hygiene.md): plugin not found, binaries overwritten, tuned paths never built → prevented | ✓2 ✗0
- [ship-prebuilt-engine](env/ship-prebuilt-engine.md): engine built at start or in CI → cold start about halved | ✓4 ✗0
- [engine-build-variance](env/engine-build-variance.md): rebuilds differ in speed and outputs → identical or best of N | ✓3 ✗0
- [trt-builder-level](env/trt-builder-level.md): prototype engine slower than production → -14% | ✓1 ✗0
- [torch-export-onnx](env/torch-export-onnx.md): deprecated TorchScript exporter → speed equal, +390 MiB on the L20, L1 | ✓1 ✗1
