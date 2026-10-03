# Boundaries and modes

## Choose the boundary

Compile the smallest stable callable that contains the main measured cost. Inputs and outputs should be tensors or structurally stable PyTrees. Keep Python I/O, JPEG bytes, PIL/OpenCV objects, logging, request metadata, weight loading, mutable callbacks, and service error handling outside.

Compile after `eval()`, device placement, dtype selection, and weight loading; afterwards do not change parameters, buffers, hooks, adapters, device, or dtype. Use `torch.inference_mode()` exactly as the eager path does. For a VAE, compile only the inner `encode` or `decode`; the outer layer may validate shapes and route buckets, but must not catch an error and call eager.

## Mode ladder

Start from explicit defaults:

```python
compiled = torch.compile(candidate, backend="inductor", mode="default", fullgraph=True, dynamic=False)
```

When the boundary is expected to capture completely, start with `fullgraph=True` so unsupported Python or operators surface as failures. Do not relax it right away and mistake partial compilation for a complete result; evaluate a partial graph only after reporting where it breaks and what that costs end to end.

Evaluate each mode in its own fresh process:

| Mode | When | Check |
| --- | --- | --- |
| `default` | first candidate of every approved experiment | compile time, warm latency, graph variants, peak memory |
| `reduce-overhead` | launch or Python overhead is material and shapes satisfy CUDA Graphs | static addresses, recapture, concurrency, peak memory, output lifetime |
| `max-autotune` | GEMM/convolution heavy, long builds and a controlled cache are acceptable | tuning time, cache portability, startup SLA, warm end to end |
| `max-autotune-no-cudagraphs` | tuning helps but CUDA Graphs do not fit | confirm the pinned PyTorch version supports the mode |

Do not assume `default` always uses less memory or `reduce-overhead` is always fastest; measure the common and maximum shapes on the target PyTorch/CUDA/GPU combination.

## CUDA Graph interaction

`reduce-overhead` may use CUDA Graphs to cut Python launch overhead. Before enabling it, verify that shapes and control flow are stable, the capture region has no CPU synchronization, custom operators are compatible, and input/output lifetimes are safe; measure the graph memory pool and the first capture of every shape.

Unless explicitly designed and tested, do not wrap a manual `torch.cuda.CUDAGraph` around code that also relies on a compile mode's CUDA Graphs. Multiple workers or concurrent streams need an explicit ownership model; single-stream serial benchmarks do not prove them safe.

## Production backend choice

Eager and compiled deployments are explicit configurations, used for offline comparison and release rollback. A compiled process that violates its contract must fail at startup warmup or during the request; never `try/except` inside a request and call eager.
