---
tier: once
read_when: first sessions in a new project (STATUS.md phase is onboarding)
---
# Onboarding a project

Goal: agree with the user on what "faster" means here, then measure the unmodified project precisely.
The user's review matters most in this phase: a misunderstanding here misdirects every later experiment.

Copy this checklist into your reply and tick it off as you go:

    - [ ] 1 Setup  - [ ] 2 Read  - [ ] 3 Discuss  - [ ] 4 Harness  - [ ] 5 Baseline

## 1. Setup
- If the project is not a git repo, ask the user. With their okay, run `git init` and commit the code as delivered; the baseline needs a commit, and nothing is pushed.
- If lx-work/ is missing, run `lx-e2espeed/setup.sh project` from the project root once the user agrees. It adds both lx directories to `.git/info/exclude`, creates lx-work from `templates/work-skeleton/`, and makes lx-work a git repo.
- Check that `git -C lx-e2espeed remote get-url origin` is the main copy on this machine; wrap-up syncs rule changes through it.
- The rest of lx-work is built in steps 2 to 5, following the lx-work layout in CLAUDE.md.

## 2. Read the project
- Trace the end-to-end path: input → preprocessing → host-to-device copy → model → device-to-host copy → postprocessing. Follow that path and skip unrelated code; a large repo does not need a full read.
- Record the path in MAP.md as `path:function` pointers, with an ASCII flow of the stages for a multi-stage pipeline.
- Note for each stage its device, dtype, layout, and shape, and every device or representation change between stages; note normal and maximum shapes, batch policy, and ROI or tile behavior. Label likely bottlenecks as hypotheses until measured.
- Find how the project runs: its Python environment (for example a conda env), the entry command, and where the weights live. Call the interpreter by its absolute path; `conda run` and a polluted PATH can pick another Python.
- Write code issues and optimization leads you notice into BACKLOG.md, each with its evidence.

## 3. Discuss with the user
Explain your understanding first, then settle each item. Ask in plain language, one or two items at a time, each with your proposed answer, so the user can reply with a yes or one correction; never hand over the whole list at once. Items the code already answers need only a confirmation.
Record the answers in PROJECT.md and each decision with its reason in DECISIONS.md. They belong to this project only; lx-e2espeed never records them.
- End-to-end scope: where timing starts and stops.
- Workload: input shapes, batch sizes, request pattern, warm or cold start.
- Metric: latency by default; throughput or memory only if the user says so.
- Accuracy standard: how outputs are compared, the metrics and pass criterion, the thresholds of difference levels L0 to L3 (see `rules/benchmarking.md`), and which level is acceptable.
- Contracts: the public interface downstream code relies on (argument order, return structure, channel order, sizes), the pre/post tensor contract (dtype, range, layout, color order, interpolation, rounding), and who consumes the output. Changing any of them needs the downstream impact stated first.
- Deployment: how production runs the code (image or environment and versions, how it loads the repo and resolves weights, workers per GPU, MPS), the maximum input it accepts, and what a delivery must include (CI steps that rebuild engines, dependency locks, runtime libraries). Align the local environment with the production versions where possible.
- Done when: the target, a point of diminishing returns (for example three experiments in a row inside the noise floor), or a deadline.
- Target hardware: where the model runs in production. If that is not this machine, gains here may not transfer; say so and agree how to confirm them.
- Constraints: code that must not change, allowed dependencies, techniques the user rules out (for example pinned memory), and the private terms (project, company, product, and model names) that the checker keeps out of lx-e2espeed.
- Environment: which Python environment the tools run in. Installing anything into it counts as a new dependency. Read `hardware/rtx-5090.md` and note what matters.

Move on only after the user confirms.

## 4. Build the harness
Read `rules/benchmarking.md` first.
- Check that the profilers run (`nsys --version`, `ncu --version`, then one `ncu --set full` run). Both ship in the CUDA toolkit's bin/, which may not be on PATH, and profiling permissions can change after a container restart; record the working commands in PROJECT.md.
- If CUDA looks unavailable, check whether the command ran in a sandbox before concluding the GPU or driver is missing, and say which it was.
- Write `lx-work/tools/check-env/`: one command that lists every gap at once (modules, CUDA in the framework, compiler, TensorRT headers matching the runtime, engines loadable, current GPU). Run it first whenever the environment misbehaves.
- `lx-work/adapter/`: load the model, build inputs, run once end to end. Bench and verify both use it, so every tool measures the same path. It can load a candidate from `exp/NNN-name/candidate/` in place of the project's class, so baseline and candidates run through the same tools. Tools take the project directory as an argument, defaulting to the project root, so they also run on a copy from `exp.sh export`.
- `lx-work/tools/`: create only what this project needs, usually bench/ and verify/. List each tool in tools/INDEX.md.

## 5. Baseline
- The project tree must be clean: `git status --porcelain` prints nothing. Ask the user to commit or stash their own changes; untracked environment files such as weights go into `.git/info/exclude` with their okay. `exp.sh` sees only this repo's own files: if a change must touch a git submodule, ask the user first.
- Record `git rev-parse --short HEAD` as Baseline commit in PROJECT.md. `exp.sh` builds every experiment on it.
- Run verify and bench on the unmodified project code.
- Measure the noise floor and the time breakdown as `rules/benchmarking.md` describes. Record the noise floor in PROJECT.md and the breakdown in MAP.md.
- Save numbers, reference outputs, and an environment snapshot (`pip freeze` or `conda list --export`) in lx-work/baseline/. Reference outputs over 50 MB go to lx-work/local/; commit their checksum and the command that regenerates them.
- Fill the baseline row in STATUS.md, set the phase to `optimizing`, and commit lx-work.
- From here on the reference outputs are frozen.
