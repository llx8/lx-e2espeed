# CUDA Memory Snapshot Workflow

## Generate a snapshot

Use the standard PyTorch sequence around the real code under observation:

```python
torch.cuda.memory._record_memory_history()

run_your_code()

torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
```

Use `scripts/cuda_memory_snapshot.py` for a reusable wrapper. Start history before allocations relevant to the suspected peak. On OOM, dump before destroying the process state when possible.

After dumping, explicitly tell the user:

- the absolute `.pickle` path;
- the exact input dimensions and repetition;
- whether the snapshot came from OOM or high-peak diagnosis;
- that the file contains allocation history and tracebacks and should be retained for analysis.

The memory-history and snapshot APIs are private PyTorch APIs. Keep their use in diagnostics, pin the working PyTorch environment, and do not place always-on recording in the production request path.

## Read the peak

Identify the peak event and inspect allocations still live there. Group them by stack trace, size, stage, dtype, and lifetime. Distinguish:

- persistent weights and caches;
- TensorRT context/workspace;
- model activations and outputs;
- full-resolution preprocessing/postprocessing temporaries;
- tile lists or accumulators kept alive together;
- copies caused by dtype, layout, device, or representation conversion;
- references retained across E2E repetitions.

Correlate allocation stacks with stage timing and explicit NVTX/record-function ranges. `reserved` growth alone is not proof of a leak; live allocation growth across identical repetitions is stronger evidence.

## Choose the fix

For full-resolution image operations, eliminate duplicate tensors first, then use ordinary or overlap tiles according to neighborhood requirements. Allocate the output canvas once and process bounded tiles sequentially.

For large VAE encode/decode, enable Diffusers VAE tiling and preserve its overlap/blend semantics. If the VAE remains a measured latency hotspot, route its fixed tile encoder/decoder to the `gpu-inference-cnn-tensorrt` skill and reuse shared TensorRT context memory where execution is non-concurrent.

For retained references, shorten lifetime and remove containers or closures holding request tensors. For workspace duplication, use the shared TensorRT memory pool only when contexts cannot overlap.

Always validate seams, edges, mask behavior, final image quality, peak memory, and repeated full E2E stability after the change.
