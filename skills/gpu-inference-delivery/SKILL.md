---
name: gpu-inference-delivery
description: "Handoff checks before delivering an optimized inference service: fail fast, self-contained repo, weight paths and store, runtime libraries."
---

# Deliver an optimized inference service

Which checks apply depends on how this project is deployed; the answers belong in `lx-work/PROJECT.md` from onboarding. Ask the user for anything the repository cannot tell, such as a remote weight store path, and mark the delivery blocked until you have it. Never assume a default mount point.

## Always

1. **Fail fast.** The selected production backend either completes or stops with an actionable error. Show that an initialization or execution failure terminates instead of silently running a slower path (PyTorch, CPU, PIL/OpenCV, a smaller model). Optional-dependency detection belongs in setup or diagnostics, never per request.
2. **Reference path kept, not used as fallback.** The eager or reference implementation stays available as an explicit offline configuration for comparison and rollback.
3. **Self-contained.** Delivered code imports nothing from `lx-work/`, `lx-e2espeed/`, sibling repositories, `/tmp`, or build directories. See [references/handoff-checklist.md](references/handoff-checklist.md) section 3 for the scans.
4. **The build path contains the change.** An optimization that exists only in a local working tree, a symlinked copy, or an uncommitted export patch is not delivered. Check dependency locks, export scripts, and CI steps that rebuild engines, and name what must be committed or re-locked.
5. **Engines are not built in the request path.** Check engine, plugin, dtype, shape, and path mismatches fail fast at startup.

## When the contract calls for it

- **Weight path contract**: when production loads the algorithm repo as a subdirectory (for example from a service wrapper), every weight is referenced relative to the repo root and joined in code; test by starting the entry point from another directory. Section 1 of the checklist.
- **Remote weight store**: when weights are deployed to a store, inventory them with SHA-256, compare, sync what differs, re-verify, and scan for symlinks. Section 2.
- **Shared runtime libraries**: when the service loads TensorRT plugins or other `.so` files, they ship with the project and are recorded with GPU architecture, CUDA, TensorRT ABI, source revision, and SHA-256. Section 3.
- **Multi-worker GPU sharing**: when several workers share a GPU through MPS, scan engines with the `gpu-inference-mps-hang` skill first.
- **Maximum input**: when the service accepts very large inputs, the repeated maximum-input run from the `gpu-inference-memory` skill has passed.

## Facts to state at delivery

Baseline environment, commands, inputs, warmup, and repetitions; before/after end-to-end latency (cold start and warm P50/P90/P99) and peak memory; each affected stage's latency and share of the original total; the complete accuracy command, input matrix, metrics, tolerances, and result; the pre/post tensor contract and which semantics were verified unchanged; files and runtime artifacts changed; assumptions, unsupported shapes, cold-start cost, and the rollback path; what was not measured or validated; the remaining bottleneck and the next evidence-backed candidate. Never infer a speedup from code shape alone.
