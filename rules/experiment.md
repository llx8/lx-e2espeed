---
tier: on-demand
read_when: phase is optimizing; before proposing, running, or reviewing an experiment
---
# Running an experiment

One experiment answers one question with one hypothesis, recorded in `lx-work/exp/NNN-name/`. It may compare several variants of one idea, but at most one lands in the project.
The user's review is worth most at the proposal and the plan: a wrong plan costs more than a wrong line of code.

## 1. Find where the time goes
- Start from the time breakdown in MAP.md and profile the largest share. A step that takes 5% of the time can save at most 5% (Amdahl's law).
- Use a timeline profiler (Nsight Systems) for the whole path: CPU gaps, launches, copies, syncs. Use a kernel profiler (Nsight Compute) only on a kernel that dominates. Raw profiles go to lx-work/local/profiles/.
- An `ncu` report collects SpeedOfLight, the roofline chart, compute and memory workload analysis with tables, occupancy, scheduler and warp-state statistics, and source counters, with `--import-source yes --import-sass yes`, and one launch per distinct kernel type rather than every repeat.
- Route by measured behavior, not by model name, and load the matching skill:

| Evidence | Skill |
|---|---|
| no trustworthy measurement yet, noisy numbers, cold start mixed into latency | `perf-workload-profiling` |
| timeline or single-kernel analysis | `perf-nsight-systems`, `perf-nsight-compute-analysis` |
| `.cpu()`, `.numpy()`, PIL/OpenCV round trips, syncs, repeated resize or cast, empty work | `perf-torch-sync-free` |
| an eager path bound by kernel launches | `perf-torch-cuda-graphs` |

- No skill covers memory, TensorRT, diffusion, torch.compile, or MPS (the user retired our own skills, 2026-10-13): start from the matching section of `playbook/INDEX.md`. When several workers share a GPU through MPS, `playbook/serving/mps-grid-barrier.md` applies from the first experiment.
- Avoid the classic anti-methods: changing things at random, looking only where the tools make it easy, and benchmarking without a question to answer.

## 2. Propose
- Parked trials in `lx-work/trials/INDEX.md` are proposals already measured; `rules/self-iteration.md` says how to adopt one.
- Read the matching category in `playbook/INDEX.md`, then only the matching cards, and `lx-work/BACKLOG.md`. A card's "Does not apply" lists conditions from past projects: check whether they hold here, and do not skip a technique only because it failed elsewhere.
- Give the user one to three options as a short table: change, evidence, expected gain (bounded by the step's share), expected difference level, risk, effort.
- Default order, which evidence may change: measurement correctness; redundant work, CPU/GPU round trips, syncs, and repeated initialization; peak memory and large-input routing; mature backends such as TensorRT, or Nunchaku for a supported diffusion transformer with weights the user supplies already quantized (a mismatch fails before inference; full precision stays an explicit offline reference); `torch.compile`, only after explicit approval; custom kernels, only for a measured hotspot with no adequate mature implementation.
- Propose a technique only with evidence that it applies here, never because it is commonly listed.
- Keep tensors on the GPU from the first boundary worth moving to the last: the model takes a CUDA tensor, pre- and postprocessing stay on the GPU, in FP16 where the pre/post contract in PROJECT.md allows, and only compact results (masks, boxes, metadata) come back to the CPU. A cheap stage moves only when the transfer it removes is measured.
- Prefer a mature GPU library (PyTorch, torchvision, Kornia, cuCIM, nvImageCodec) that expresses the reference semantics within the agreed tolerance over a custom kernel, which needs a measured hotspot and no adequate operator. Reproducing a CPU library's incidental bit-level behavior at a material cost is the user's call, as is the tolerance of a cheaper approximation of a simple operation (a box-blur chain for a large Gaussian, repeated erosion for an elliptical kernel): propose it with its measured difference.
- A `torch.compile` proposal is an assessment, with no compile call, Inductor or Dynamo setting, or cache variable tried: the measured hotspot, the module boundary (never an HTTP handler, codec, or loader), the shape distribution and the policy (static, buckets keyed by the shape attributes, or bounded dynamic with `mark_dynamic` ranges), the mode, the warmup and compile budget, the cache identity for deployment (`TORCHINDUCTOR_CACHE_DIR` set by the deployment config; cold starts with an empty, a warm, and an incompatible cache; the budget times the worker count), the expected gain, the accuracy plan, and the rollback, marked "not enabled; needs explicit approval".
- Propose only work on the path production runs (PROJECT.md, Deployment; knowledge/deployment.md). When the user says production calls the code another way, re-check the MAP.md path first: a step the bench times but production skips is no gain, so say so instead of proposing it.
- "Nothing left to propose" is a measured claim: re-profile best and close each lead in BACKLOG.md with a measurement, not an estimate, before making it. One run stopped early with about 40 us a call unexplained.
- When the path forks on a user judgment, such as whether a difference level is acceptable, measure the options on each side of the fork so the user decides with numbers. Repeat at each fork until no option is left: a method tree.
- The user picks one, or none.

## 3. Plan
- Run `lx-e2espeed/exp.sh new <1 to 4 words>`. It checks that the project tree equals the baseline commit plus the kept patches, and each linked repo (PROJECT.md, `- Linked repos:`) its commit there plus the kept patches, takes the next number, creates the directory with NOTE.md, and adds a running row to exp/INDEX.md.
- Write Evidence, Hypothesis, and Plan in NOTE.md: the functions to change, the contracts in PROJECT.md it touches and their downstream impact, how verify covers it, and the expected gain.
- Show the plan to the user and build nothing until they approve it. For a change that fits in one sentence and touches no contract, that sentence is the plan.
- Refactor first when landing would otherwise bend the code: the change would be the third kept patch on the same function, or it needs a structure the code lacks (a stage boundary, a home for a table or a kernel). The refactor is its own experiment, with verify and a bench that shows equal speed, so the next change lands as a replacement instead of another layer. Judge this when a plan calls for it, not after every experiment: a refactor between kept patches costs its own verify and bench, and a question asked every time gets skipped.

## 4. Build and compare candidates
- Write each candidate in `lx-work/exp/NNN-name/candidate/`, as a copy, subclass, or hook around the project code; use one subdirectory per variant when comparing several. The project code stays untouched while exploring, so best and the candidates run side by side.
- Bench each variant against best (the A/B, `--out lx-work/exp/NNN-name/ab-<variant>.json`, since the next A/B overwrites the per-round files) and verify it against the frozen references, as `rules/benchmarking.md` describes.
- Report every variant to the user as three sets per scenario, from the bench's summary lines: baseline, before (best, the A/B's best side), and after (the candidate), each with median, P90, P99, and memory, plus the difference level. The baseline comes from `BASELINE` in the adapter, never from the baseline commit's bench unless that is the baseline; its change is descriptive (`rules/benchmarking.md`, Inputs).
- Build experimental binaries into their own build directory, never over production ones. Compiled libraries are not bit-reproducible, so compare builds by behavior, never by hash.
- After approval, compile the smallest hotspot first, from `backend="inductor", mode="default", fullgraph=True, dynamic=False`; `reduce-overhead` only for a launch-bound path within CUDA Graph constraints, `max-autotune` only when its build time is acceptable. Each candidate runs in a fresh cold process over every shape bucket, recording compile time, first call, warmup convergence, the hot distribution, peak memory, graph breaks and recompiles (`TORCH_LOGS="graph_breaks,recompiles,guards"`, `torch._dynamo.explain`), cache hits, and end to end. Compiled and eager are two explicit deployment configurations, never a try/except fallback; compiler state is never reset in the request path; reject when the end-to-end value is below the cold-start, memory, cache, and maintenance cost.
- A deviation from the plan goes back to the user first. If verify fails, fix the candidate or drop it; never loosen the check.

## 5. Land and decide
- The user picks a variant, or none.
- To land it, apply the smallest change to the project code that reproduces the chosen variant, keeping file paths, interfaces, and contracts. Verify and bench again on the real code, save the exact commands as run.sh and the bench output as result.json, then run `exp.sh keep`: it saves change.patch, plus change.<dir>.patch for each linked repo the change touched, and checks that the kept patches rebuild the tree.
- Land a replacement, not an addition, so the kept patches do not pile every stage of the project into one file: the new path takes the place of the old one, and code the new path makes unreachable goes, unless PROJECT.md asks for a fallback; a switch with one production value goes with it; derivations and measurements go to the project's docs or the experiment's NOTE.md, not into comments around the function; timing prints, variant switches, and other candidate scaffolding stay in candidate/. The old code is still in the baseline commit and in the patches, so deleting it loses nothing.
- To drop, run `exp.sh drop`: it reverses anything the experiment changed in the project and its linked repos, and checks that the tree is back. Never undo by hand.
- Both warn about files the project ignores (build outputs, caches) that the experiment wrote. No patch holds them: run.sh must rebuild them, and after a drop remove the ones the experiment made, so they cannot change the next bench.
- A gain inside the noise floor is not a gain. A small gain that adds a lot of complexity may not be worth keeping; deleting code at equal speed is a win.
- Fill in Result and Conclusion in NOTE.md, and the numbers, difference level, and reason in exp/INDEX.md. Record the A/B's numbers everywhere (the three sets in NOTE.md, the median and its changes in exp/INDEX.md); the bench of the landed code only confirms them. A dropped experiment keeps its record and says what was learned; knowing what failed saves time later. If the user rules out a whole direction, record it in DECISIONS.md with the reason, so it is not proposed again in this project.
- If the kept patch changed a file MAP.md points to, re-run the stage timing on best and update the breakdown, the hot spots, and `- Measured on:` in MAP.md. The checker holds the lx-work commit until then: shares are relative, so every landed gain moves all of them.
- Update STATUS.md and BACKLOG.md, and make the experiment's closing lx-work commit, kept or dropped. Its records are also committed as they are written (plan, candidates, results), with "In progress" in STATUS.md current: records left uncommitted for hours are lost to a crash, and the next session cannot see them.
- If the done-when condition in PROJECT.md is met, tell the user and suggest wrapping up.

## When the project changes underneath
`exp.sh new` refuses when the project HEAD is not the baseline commit or the tree differs from the baseline commit plus the kept patches, and likewise for each linked repo and its commit in PROJECT.md. Tell the user; do not repair it yourself.
When the user committed exactly the kept patches, `exp.sh delivered` makes that commit the new baseline commit, and each linked repo's HEAD its new commit in `- Linked repos:`.
Moving to newer company code needs a new baseline commit or linked commit, only with the user's approval, recorded in DECISIONS.md: set it in PROJECT.md, bring each kept patch forward until `exp.sh check` passes, then verify, bench, and re-run the stage timing for MAP.md again.
Either way the baseline (speed and reference outputs) stays; only the user changes it, recorded in DECISIONS.md.

## Reproducing or reverting
- `exp.sh export NNN` writes the code of any experiment (its baseline commit, the kept patches before it, and its own patch) to lx-work/local/replay/, each linked repo in its own subdirectory; run its run.sh against that copy. `exp.sh export best` and `baseline-commit` do the same for A/B benches.
- To take back a kept experiment, run a new experiment whose change reverses it (`git apply -R` of its patches), so the kept patches stay one straight line.
- Files deleted from lx-work stay in its history: `git -C lx-work log --diff-filter=D --name-only`.
