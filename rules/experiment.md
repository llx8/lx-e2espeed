---
tier: on-demand
read_when: phase is optimizing; before proposing, running, or reviewing an experiment
---
# Running an experiment

One experiment answers one question with one hypothesis, recorded in `lx-work/exp/NNN-name/`. It may compare several variants of one idea; at most one lands.
The user's review is worth most at the proposal and the plan: a wrong plan costs more than a wrong line of code.

## 1. Find where the time goes
- Start from the time breakdown in MAP.md and profile the largest share: a step that takes 5% of the time can save at most 5%.
- Nsight Systems for the whole path (CPU gaps, launches, copies, syncs); Nsight Compute only on a kernel that dominates, collecting SpeedOfLight, roofline, compute and memory workload tables, occupancy, scheduler and warp-state statistics, and source counters (`--import-source yes --import-sass yes`), one launch per distinct kernel type. Raw profiles go to lx-work/local/profiles/.
- Route by measured behavior, not by model name, and load the matching skill:

| Evidence | Skill |
|---|---|
| no trustworthy measurement yet, noisy numbers, cold start mixed into latency | `perf-workload-profiling` |
| timeline or single-kernel analysis | `perf-nsight-systems`, `perf-nsight-compute-analysis` |
| `.cpu()`, `.numpy()`, PIL/OpenCV round trips, syncs, repeated resize or cast, empty work | `perf-torch-sync-free` |
| an eager path bound by kernel launches | `perf-torch-cuda-graphs` |

- Memory, TensorRT, diffusion, torch.compile, and MPS have no skill: start from the matching section of `playbook/INDEX.md`. When several workers share a GPU through MPS, `playbook/serving/mps-grid-barrier.md` applies from the first experiment.
- Avoid changing things at random, looking only where the tools make it easy, and benchmarking without a question.

## 2. Propose
- Parked trials in `lx-work/trials/INDEX.md` are proposals already measured (`rules/self-iteration.md` says how to adopt one).
- Read the matching category in `playbook/INDEX.md`, then only the matching cards, and `lx-work/BACKLOG.md`. A card's "Does not apply" lists past conditions: check whether they hold here; a technique that failed elsewhere is not skipped for that alone.
- Give the user one to three options as a short table: change, evidence, expected gain (bounded by the step's share), expected difference level, risk, effort. Only techniques with evidence that they apply here, only on the path production runs (PROJECT.md, Deployment): a step the bench times but production skips is no gain, so say so.
- Default order, which evidence may change: measurement correctness; redundant work, round trips, syncs, repeated initialization; peak memory and large-input routing; mature backends such as TensorRT, or Nunchaku for a supported diffusion transformer with weights the user supplies already quantized; `torch.compile` only after explicit approval; custom kernels only for a measured hotspot with no adequate mature implementation.
- Keep tensors on the GPU from the first boundary worth moving to the last: the model takes a CUDA tensor, pre- and postprocessing stay on the GPU (FP16 where the pre/post contract allows), only compact results come back. A cheap stage moves only when the transfer it removes is measured.
- Prefer a mature GPU library (PyTorch, torchvision, Kornia, cuCIM, nvImageCodec) that meets the reference semantics within the tolerance over a custom kernel. Reproducing a CPU library's incidental bit-level behavior at material cost, or the tolerance of a cheaper approximation (a box-blur chain for a large Gaussian), is the user's call: propose it with its measured difference.
- A `torch.compile` proposal is an assessment with nothing compiled, marked "not enabled; needs explicit approval" (`playbook/compute/torch-compile.md`).
- "Nothing left to propose" is a measured claim: re-profile best and close each BACKLOG.md lead with a measurement first.
- When the path forks on a user judgment (is this level acceptable), measure the options on each side so the user decides with numbers, at each fork: a method tree.
- The user picks one, or none.

## 3. Plan
- `lx-e2espeed/exp.sh new <1 to 4 words>` checks that the project tree and each linked repo equal their commits plus the kept patches, takes the next number, creates the directory with NOTE.md, and adds a running row to exp/INDEX.md.
- Write Evidence, Hypothesis, and Plan in NOTE.md: the functions to change, the contracts it touches and their downstream impact, how verify covers it, the expected gain.
- Build nothing until the user approves the plan; a change that fits one sentence and touches no contract has that sentence as its plan. In a standard pass the catalog is the approved plan.
- Refactor first when landing would bend the code: a third kept patch on one function, or a structure the code lacks (a stage boundary, a home for a table or a kernel). The refactor is its own experiment, verified, with a bench showing equal speed. Judge it when a plan calls for it, not after every experiment.

## 4. Build and compare candidates
- Write each candidate in `lx-work/exp/NNN-name/candidate/` as a copy, subclass, or hook around the project code, one subdirectory per variant. The project code stays untouched while exploring.
- A/B each variant against best (`--out lx-work/exp/NNN-name/ab-<variant>.json`) and verify it against the frozen references (`rules/benchmarking.md`).
- Report each variant as three sets per scenario from the bench's summary lines: baseline (from `BASELINE` in the adapter, descriptive), before (best), after (the candidate), each with median, P90, P99, and memory, plus the difference level.
- Experimental binaries go to their own build directory, never over production ones; compare builds by behavior, never by hash.
- A deviation from the plan goes back to the user first. If verify fails, fix or drop the candidate; never loosen the check.

## 5. Land and decide
- The user picks a variant, or none.
- Landing applies the smallest change that reproduces the chosen variant, keeping file paths, interfaces, and contracts. Verify and bench the real code, save the exact commands as run.sh and the bench output as result.json, then `exp.sh keep`: it saves change.patch (and change.<dir>.patch per linked repo touched) and checks that the kept patches rebuild the tree.
- Land a replacement, not an addition: code the new path makes unreachable goes, unless PROJECT.md asks for a fallback, and so does a switch with one production value. Derivations and measurements go to the docs or NOTE.md, not into comments; timing prints and variant switches stay in candidate/.
- To drop, `exp.sh drop`: it reverses the experiment in the project and its linked repos and checks the tree. Never undo by hand.
- Both warn about ignored files the experiment wrote (build outputs, caches): run.sh must rebuild them, and after a drop remove the ones it made, so they cannot change the next bench.
- A gain inside the noise floor is not a gain; a small gain with much complexity may not be worth keeping; deleting code at equal speed is a win.
- Fill Result and Conclusion in NOTE.md, and the numbers, level, and reason in exp/INDEX.md, from the A/B (the bench of the landed code only confirms them). A dropped experiment says what was learned. A direction the user rules out goes in DECISIONS.md with the reason.
- If the kept patch changed a file MAP.md points to, re-run the stage timing on best and update the breakdown, hot spots, and `- Measured on:`; the checker holds the lx-work commit until then.
- Update STATUS.md and BACKLOG.md and make the closing lx-work commit. Commit records as they are written, with "In progress" current: uncommitted records are lost to a crash.
- If the done-when condition in PROJECT.md is met, suggest wrapping up.

## When the project changes underneath
`exp.sh new` refuses when HEAD is not the baseline commit or the tree differs from it plus the kept patches, and likewise for each linked repo. Tell the user; do not repair it yourself.
When the user committed exactly the kept patches, `exp.sh delivered` makes that commit the new baseline commit (and each linked HEAD its new commit).
Moving to newer company code needs the user's approval, recorded in DECISIONS.md: set the new commit in PROJECT.md, bring each kept patch forward until `exp.sh check` passes, then verify, bench, and re-run the stage timing.
Either way the baseline (speed and reference outputs) stays; only the user changes it.

## Reproducing or reverting
- `exp.sh export NNN` writes any experiment's code (baseline commit, earlier kept patches, its own patch) to lx-work/local/replay/, linked repos in subdirectories; `export best` and `baseline-commit` do the same for A/B benches.
- To take back a kept experiment, run a new one whose change reverses it (`git apply -R`), so the kept patches stay one line.
- Files deleted from lx-work stay in its history: `git -C lx-work log --diff-filter=D --name-only`.
