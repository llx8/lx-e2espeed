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
| no trustworthy measurement yet, noisy numbers, cold start mixed into latency | `gpu-inference-profiling`, `perf-workload-profiling` |
| timeline or single-kernel analysis | `perf-nsight-systems`, `perf-nsight-compute-analysis` |
| `.cpu()`, `.numpy()`, PIL/OpenCV round trips, syncs, repeated resize or cast, empty work | `gpu-inference-dataflow`, `perf-torch-sync-free` |
| OOM, high peak, large inputs, many tiles | `gpu-inference-memory` |
| a CNN, restoration, or segmentation stage dominates with stable shapes | `gpu-inference-cnn-tensorrt` |
| an eager path bound by kernel launches | `perf-torch-cuda-graphs` |
| diffusion transformer, VAE, LoRA | `gpu-inference-diffusion` |
| an eager hotspot remains after the above | `gpu-inference-torch-compile`, only after explicit approval |
| several workers share a GPU through MPS | `gpu-inference-mps-hang` |

- Avoid the classic anti-methods: changing things at random, looking only where the tools make it easy, and benchmarking without a question to answer.

## 2. Propose
- Parked night trials in `lx-work/night/INDEX.md` are proposals already measured; `rules/night.md` says how to adopt one.
- Read the matching category in `playbook/INDEX.md`, then only the matching cards, and `lx-work/BACKLOG.md`. A card's "Does not apply" lists conditions from past projects: check whether they hold here, and do not skip a technique only because it failed elsewhere.
- Give the user one to three options as a short table: change, evidence, expected gain (bounded by the step's share), expected difference level, risk, effort.
- Default order, which evidence may change: measurement correctness; redundant work, CPU/GPU round trips, syncs, and repeated initialization; peak memory and large-input routing; mature backends such as TensorRT or an established diffusion backend; `torch.compile`, only after explicit approval; custom kernels, only for a measured hotspot with no adequate mature implementation.
- Propose a technique only with evidence that it applies here, never because it is commonly listed.
- Propose only work on the path production runs (PROJECT.md, Deployment; knowledge/deployment.md). When the user says production calls the code another way, re-check the MAP.md path first: a step the bench times but production skips is no gain, so say so instead of proposing it.
- "Nothing left to propose" is a measured claim: re-profile best and close each lead in BACKLOG.md with a measurement, not an estimate, before making it. One run stopped early with about 40 us a call unexplained.
- When the path forks on a user judgment, such as whether a difference level is acceptable, measure the options on each side of the fork so the user decides with numbers. Repeat at each fork until no option is left: a method tree.
- The user picks one, or none.

## 3. Plan
- Run `lx-e2espeed/exp.sh new <1 to 4 words>`. It checks that the project tree equals the baseline commit plus the kept patches, and each linked repo (PROJECT.md, `- Linked repos:`) its commit there plus the kept patches, takes the next number, creates the directory with NOTE.md, and adds a running row to exp/INDEX.md.
- Write Evidence, Hypothesis, and Plan in NOTE.md: the functions to change, the contracts in PROJECT.md it touches and their downstream impact, how verify covers it, and the expected gain.
- Show the plan to the user and build nothing until they approve it. For a change that fits in one sentence and touches no contract, that sentence is the plan.

## 4. Build and compare candidates
- Write each candidate in `lx-work/exp/NNN-name/candidate/`, as a copy, subclass, or hook around the project code; use one subdirectory per variant when comparing several. The project code stays untouched while exploring, so best and the candidates run side by side.
- Bench each variant against best (the A/B, `--out lx-work/exp/NNN-name/ab-<variant>.json`, since the next A/B overwrites the per-round files) and verify it against the frozen references, as `rules/benchmarking.md` describes.
- Report every variant to the user as three sets per scenario, from the bench's summary lines: baseline, before (best, the A/B's best side), and after (the candidate), each with median, P90, P99, and memory, plus the difference level. The baseline comes from `BASELINE` in the adapter, never from the baseline commit's bench unless that is the baseline; its change is descriptive (`rules/benchmarking.md`, Inputs).
- Build experimental binaries into their own build directory, never over production ones. Compiled libraries are not bit-reproducible, so compare builds by behavior, never by hash.
- A deviation from the plan goes back to the user first. If verify fails, fix the candidate or drop it; never loosen the check.

## 5. Land and decide
- The user picks a variant, or none.
- To land it, apply the smallest change to the project code that reproduces the chosen variant, keeping file paths, interfaces, and contracts. Verify and bench again on the real code, save the exact commands as run.sh and the bench output as result.json, then run `exp.sh keep`: it saves change.patch, plus change.<dir>.patch for each linked repo the change touched, and checks that the kept patches rebuild the tree.
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
