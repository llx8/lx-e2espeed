# Nsight Systems Analysis

## Contents

- Capture
- Read the timeline
- Rank TensorRT work
- Escalate

## Capture

Profile a warmed, finite TensorRT loop with NVTX ranges. Avoid engine build and model loading in the capture window.

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --stop-on-range-end=true \
  -o trt-steady \
  python run_profile.py
```

Use `torch.cuda.cudart().cudaProfilerStart()` and `cudaProfilerStop()` around the warmed repetitions when capture-range control is needed. For the bundled engine probe, omit capture-range options and filter the `tensorrt.engine.steady_state` NVTX range.

Export summaries with `scripts/tools/nsys_summary.py trt-steady.nsys-rep`.

## Read the timeline

1. Confirm the expected NVTX request and engine ranges exist.
2. Check gaps between CUDA kernels. Long CPU gaps indicate launch, locking, allocation, Python, or synchronization overhead rather than a slow TensorRT kernel.
3. Rank `cuda_gpu_kern_sum` by total time, then inspect call count and mean duration. High total time can mean one expensive kernel or excessive repetition.
4. Inspect `cuda_api_sum` for `cudaMemcpy*`, allocation, stream synchronization, event synchronization, and module loading.
5. Compare kernel sequences across shapes. Tactic or profile changes can make one shape use a different implementation.
6. Map plugin kernels and TensorRT-generated kernels back to the engine layer inspector output from `scripts/tools/trt_attn_probe.py --layers`.

## Rank TensorRT work

Classify the result as compute-dominant, launch/API-dominant, copy-dominant, synchronization-dominant, or shape/tactic-dependent. Report the top kernels with total time, calls, mean, NVTX parent, input shape, and engine identity.

## Escalate

Use Nsight Compute only for one proven dominant kernel. Do not run it over the complete service. For a plugin kernel, collect occupancy, achieved bandwidth, tensor-core utilization, warp stalls, register pressure, and shared-memory behavior against the exact production shape.
