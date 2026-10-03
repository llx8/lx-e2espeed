# Diagnostics

## Locate by layer

1. Reproduce with the smallest in-project callable and real tensor metadata.
2. Confirm eager completes and passes the full accuracy check.
3. Use `backend="eager"` offline only, to diagnose Dynamo capture; it is neither a performance backend nor a production fallback.
4. Use `backend="aot_eager"` to separate Dynamo capture from Inductor lowering.
5. Use Inductor with `fullgraph=True` to find unsupported boundaries.
6. Check graph breaks, guards, and recompiles before generated kernels.
7. Keep the real service repro, then minimize the failing case.

## Logs

Turn on focused logs in a separate diagnostic process, without changing production defaults:

```bash
TORCH_LOGS="graph_breaks,recompiles,guards" python approved_compile_experiment.py
```

Add `dynamo`, `aot`, `inductor`, `output_code`, or performance hints only for a specific problem; verbose logs are large and disturb timing, so turn them off for benchmarks.

Use `torch._dynamo.explain`, or the explain API of the pinned version, to count graphs and break reasons. Private APIs and environment variables are version-pinned diagnostic tools, not stable service interfaces.

## Common failures

| Symptom | Check | Direction |
| --- | --- | --- |
| Break at `.item()` or a data-dependent branch | scalar capture, Python control flow | move the decision outside the boundary or rewrite as supported tensor control flow |
| Image size triggers repeated compiles | shape guards, strides, pad/tile branches | explicit buckets or bounded dynamic dimensions |
| Same shape still recompiles | Python values, optional arguments, module/global state, hooks | freeze service state and normalize inputs |
| Custom CUDA/Triton operator unsupported | registration, fake/meta kernels, mutation and alias declarations | register correctly or move it outside the boundary |
| Inductor compile error | generated code and a minimal repro | fix or replace the boundary; never call eager silently |
| Correct once, corrupted on repeat | CUDA Graph input/output lifetime or mutation | disable the mode and verify the address and lifetime contract |
| Warm gain but slower end to end | compile/capture, transfers, worker startup, cache misses | reject, or restructure the deployment lifecycle |
| Peak memory regression or OOM | autotuning, graph memory pool, variants, temporary buffers | compare modes, cut graph variants, route to the memory skill |

Never set broad suppression options that turn compile failures into eager execution. Experiments must expose failures, and a compiled deployment must stop with an actionable error.
