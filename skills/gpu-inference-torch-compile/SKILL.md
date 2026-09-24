---
name: gpu-inference-torch-compile
description: Assess torch.compile for a remaining eager hotspot; run experiments only after the user explicitly approves.
---

# Assess torch.compile

Treat compilation as a low-priority, opt-in optimization.

## Approval gate

Before approval, only assess applicability, name candidate modules, analyze likely graph breaks and recompilation risk, and draft the experiment. Do not:

- add `torch.compile` calls or decorators;
- change Inductor, Dynamo, or Triton settings;
- enable compile caches or their environment variables;
- make compilation the default backend or a silent fallback;
- run a compile experiment that changes project code or deployment artifacts.

Ask the user explicitly whether to approve the proposed experiment. A general request to optimize is not approval for `torch.compile`.

## Report before approval

Report the measured hotspot, candidate module boundary, production shape distribution, static / bucketed / dynamic shape policy, intended mode, CUDA Graph applicability, warmup and compile-time budget, cache and deployment identity, likely graph breaks and guards, peak-memory risk, expected value, the full accuracy plan, and the rollback plan. Use this block:

```text
Candidate: torch.compile
Status: not enabled; needs explicit user approval
Evidence:
Expected value:
Compile / cache / dynamic-shape risk:
Proposed experiment boundary:
```

## After approval

1. Freeze the eager production baseline, environment identity, input matrix, and complete accuracy command.
2. Compile the smallest repeated hotspot that covers the main cost. Prefer a module or a pure tensor function; keep request parsing, JPEG, Python I/O, logging, and mutable service state outside the boundary.
3. Start with `backend="inductor"`, `mode="default"`, and an explicit shape policy. Do not tune private Inductor settings first.
4. Run each candidate in its own fresh process. Measure compile time, first call, warmup convergence, warm distribution, peak allocated and reserved memory, graph count, graph breaks, recompiles, cache hits, and full end-to-end latency.
5. Analyze graph breaks and guards before trying `fullgraph=False`; when the boundary should form one graph, gate it with `fullgraph=True`.
6. Compare `reduce-overhead` only when Python launch overhead is material and CUDA Graph constraints hold; compare `max-autotune` only when long builds and cache management are acceptable.
7. Cover every supported shape bucket and repeated request sequences. Success on one shape is not production validation.
8. Run the complete accuracy suite against eager under the project's unchanged pre/post tensor contract.
9. Deploy compiled and eager as two explicit configurations. Once the compiled service is selected, an initialization or execution failure must terminate, not fall back to eager.
10. Reject the result when its end-to-end value does not outweigh cold start, peak memory, cache lifetime, compatibility, and maintenance cost.

Read [references/assessment.md](references/assessment.md) for admission and rejection criteria, then only what the task needs:

- [references/boundaries-and-modes.md](references/boundaries-and-modes.md): compile boundaries, modes, `fullgraph`, CUDA Graphs.
- [references/dynamic-shapes.md](references/dynamic-shapes.md): fixed shapes, buckets, guards, dynamic dimensions, VAE image sizes.
- [references/diagnostics.md](references/diagnostics.md): graph breaks, recompiles, logs, minimal repros, fail-fast diagnosis.
- [references/cache-and-deployment.md](references/cache-and-deployment.md): cold start, caches, compatibility identity, workers, immutable deployment.
- [references/benchmark-and-accuracy.md](references/benchmark-and-accuracy.md): experiment matrix, metrics, peak memory, concurrency, the accuracy gate.
