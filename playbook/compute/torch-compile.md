---
tier: on-demand
read_when: profiling shows the symptom below
---
# torch.compile on large eager models

- Category: compute
- Applies to: eager PyTorch models where a mature backend is not an option; only with explicit approval

## Symptom
Eager model stages dominate and TensorRT or another mature backend is not available for them.

## Technique
A proposal is an assessment with nothing compiled or configured, marked "not enabled; needs explicit approval" (CLAUDE.md, Hard rules). It states the measured hotspot; the module boundary (never an HTTP handler, codec, or loader); the shape distribution and the policy (static, buckets keyed by the shape attributes, or bounded dynamic with `mark_dynamic` ranges); the mode; the warmup and compile budget;
the cache identity for deployment (`TORCHINDUCTOR_CACHE_DIR` set by the deployment config; cold starts with an empty, a warm, and an incompatible cache; the budget times the worker count); the expected gain, the accuracy plan, and the rollback.
After approval, compile the smallest hotspot first, from `backend="inductor", mode="default", fullgraph=True, dynamic=False`; `reduce-overhead` only for a launch-bound path within CUDA Graph constraints, `max-autotune` only when its build time is acceptable.
Each candidate runs in a fresh cold process over every shape bucket, recording compile time, first call, warmup convergence, the hot distribution, peak memory, graph breaks and recompiles (`TORCH_LOGS="graph_breaks,recompiles,guards"`, `torch._dynamo.explain`), cache hits, and end to end.
Compiled and eager are two explicit deployment configurations, never a try/except fallback; compiler state is never reset in the request path. Reject when the end-to-end value is below the cold-start, memory, cache, and maintenance cost.

## Effect
One case: -1.3% end to end, within the stage noise of that time.

## Side effects
First compilation took about 100 s; cold start, cache identity, and maintenance.

## Does not apply
- When a TensorRT path exists: it gave 2.9x on the same model.
- When cold start matters or shapes vary per request.

## Evidence
- ✗ 2026-08: two-network image color-transfer model at 1536x1536 with D=16 attention, RTX 5090, TensorRT 10.16: 2316 -> 2285 ms, gain swallowed by variance in another stage; not adopted.
