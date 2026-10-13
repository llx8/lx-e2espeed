---
tier: once
read_when: first sessions in a new project (STATUS.md phase is onboarding)
---
# Onboarding a project

Goal: agree with the user on what "faster" means here, then measure the unmodified project precisely.
The user's review matters most in this phase: a misunderstanding here misdirects every later experiment.

Two steps, each started once the user says so. Copy this line into your reply and tick it off as you go:

    - [ ] 1 Set up and read  - [ ] 2 Discuss, then the baseline

## 1. Set up and read
Setup:
- If the project is not a git repo, ask the user. With their okay, run `git init` and commit the code as delivered; exp.sh needs a baseline commit, and nothing is pushed.
- A project starts using these rules with `lx-e2espeed/setup.sh project`, run from the project root on the user's word; until lx-work/ exists, the SessionStart hook injects nothing there. It adds lx-work/ to `.git/info/exclude`, creates lx-work from `templates/work-skeleton/`, makes lx-work a git repo, and registers the project, so the checker keeps its private terms out of lx-e2espeed and the workspace lists it (`rules/workspace.md`).
- The rest of lx-work is built in these two steps, following the lx-work layout in `rules/naming.md`.

Read the project:
- Read `lx-e2espeed/local/site.md` first when it exists: this machine's facts shared across the company's projects (images, CI, service framework, tested libraries, reference code for the usual changes), each pointing to the record it came from. It is never committed.
- Trace the end-to-end path: input → preprocessing → host-to-device copy → model → device-to-host copy → postprocessing. Follow that path and skip unrelated code; a large repo does not need a full read.
- Record the path in MAP.md as `path:function` pointers, with an ASCII flow of the stages for a multi-stage pipeline.
- Note for each stage its device, dtype, layout, and shape, and every device or representation change between stages; note normal and maximum shapes, batch policy, and ROI or tile behavior. Label likely bottlenecks as hypotheses until measured.
- Find how the project runs: its Python environment (for example a conda env), the entry command, and where the weights live. A weight file of a few hundred bytes is a Git LFS pointer: fetch the real files (`git lfs pull`) before measuring. Call the interpreter by its absolute path; `conda run` and a polluted PATH can pick another Python.
- For a diffusion pipeline, record the base model, the quantization variant of the weights, the pipeline and scheduler, the LoRA formats and ranks, and the reference workflow (for example a ComfyUI graph), which is the specification; fusing a LoRA or changing its rank is a contract change.
- Write code issues and optimization leads you notice into BACKLOG.md, each with its evidence.
- Write project, company, product, and model names into Private terms in PROJECT.md as you meet them, so the checker keeps them out of lx-e2espeed from the start. They stay provisional until the user confirms them in step 2; say so under "Waiting for the user" in STATUS.md.
- Write the rest of what the reading teaches into `lx-work/knowledge/` as you read, in the format its INDEX.md gives: who calls the code and what the outputs mean (business.md), the exact input and output formats (contract.md), and how production builds and runs it (deployment.md). After a compaction or in a new session, read these files instead of the code again.

Build the harness, so the discussion in step 2 has numbers to look at. Read `rules/benchmarking.md` first.
- Check that the profilers run (`nsys --version`, `ncu --version`, then one `ncu --set full` run). Both ship in the CUDA toolkit's bin/, which may not be on PATH, and profiling permissions can change after a container restart; record the working commands in PROJECT.md, in backticks or an indented block, where metavariables such as `<name>` are allowed.
- If CUDA looks unavailable, check whether the command ran in a sandbox before concluding the GPU or driver is missing, and say which it was.
- Write `lx-work/tools/check-env/`: one command that lists every gap at once (modules, CUDA in the framework, compiler, TensorRT headers matching the runtime, engines loadable, current GPU). Run it first whenever the environment misbehaves.
- Copy `templates/tools/adapter/` to `lx-work/adapter/` and the generic `bench/`, `verify/`, and `cpu-profile/` to `lx-work/tools/`, then fill every TODO the code answers: scenarios, stages, loading, one call, output arrays. Leave the difference levels, the time budget, and `BASELINE` for step 2. The generic tools already follow `rules/benchmarking.md` (A/B and A/A modes, stage mode, GPU sampling during the run, file-descriptor-level log redirect, self-test) and load candidates the same way as the project's code. Copy `run-image/` too when a number must come from a production image (the baseline speed in the previous image), and `engine-noise/` when the model runs as a built engine (L1 is the spread of rebuilds); each says what it needs from the adapter.
- Bench the unmodified code once, as a first number for the discussion; it is not the baseline until step 2 has set the conditions.
- Change a generic tool only for a real project need, and bring a general fix back to the templates at wrap-up. Add project-only tools as needed; list every tool in tools/INDEX.md. Keep a new tool to what answers its question; before it grows past that, tell the user what it adds and what it costs in lines.

## 2. Discuss, then the baseline
Explain your understanding first, keeping the issues found in step 1 apart from the items to decide, and say that nothing changes until the user agrees after a discussion. Then take one or two items at a time in plain language: what the item means, its options, and which you would pick and why; the user decides. Never hand over the whole list at once, and never ask for a quick yes. Items the code already answers need only a confirmation.
Record the agreed parameters in PROJECT.md, each decision with its reason in DECISIONS.md, and what the user explains about callers, contract, or deployment in the knowledge files, where an answer replaces its open question. They belong to this project only; lx-e2espeed never records them.
- End-to-end scope: where timing starts and stops.
- Workload: input shapes, batch sizes, request pattern, warm or cold start.
- Metric: latency by default; throughput or memory only if the user says so. Every result is compared with the baseline (glossary in `rules/writing.md`): its speed is the baseline commit's own bench unless the user names another source, such as production's speed in its previous environment.
- Bench time budget: how long one bench run may take. It sets WARMUP and REPEAT in the adapter and goes under Metric in PROJECT.md.
- Accuracy standard: where the reference outputs, the baseline's outputs, come from (the baseline commit's own by default), how outputs are compared, the metrics and pass criterion, the thresholds of difference levels L0 to L3 (see `rules/benchmarking.md`) or none, and which level is acceptable.
- Contracts: the public interface downstream code relies on (argument order, return structure, channel order, sizes), the pre/post tensor contract (dtype, range, layout, color order, interpolation, rounding), and who consumes the output. Changing any of them needs the downstream impact stated first.
- Deployment: how production runs the code (image or environment and versions, how it loads the repo and resolves weights, workers per GPU, MPS), the maximum input it accepts, and what a delivery must include (CI steps that rebuild engines, dependency locks, runtime libraries). Align the local environment with the production versions where possible. MPS is on by default in the user's production, and the service image starts its daemon by default: assume it unless the user says otherwise, so the MPS hazard in `playbook/serving/mps-grid-barrier.md` applies from the first experiment. A service repo that pins a separate algorithm repo in a dependency file ends its delivery with a merge and a new pin (`rules/wrap-up.md`, step 1).
- Done when: the target, a point of diminishing returns (for example three experiments in a row inside the noise floor), or a deadline.
- Target hardware: where the model runs in production. If that is not this machine, gains here may not transfer; say so and agree how to confirm them.
- Constraints: code that must not change, allowed dependencies, techniques the user rules out (for example pinned memory), and the private terms filled in provisionally in step 1: confirm them, and add any the user names.
- Environment: which Python environment the tools run in. Installing anything into it counts as a new dependency. Read the `hardware/` doc of the production GPU (`hardware/rtx-5090.md`, `hardware/l20.md`) and note what matters.

Once the user has confirmed, the baseline follows under the agreed conditions; it ends the step and the phase.
- Put the agreed difference levels and time budget into the adapter. The project tree must be clean: `git status --porcelain` prints nothing. Ask the user to commit or stash their own changes; untracked environment files such as weights go into `.git/info/exclude` with their okay. `exp.sh` sees only this repo's own files and its linked repos: git checkouts at the project top that the project ignores, such as a sibling algorithm repo or its symlink, which experiments may change. Link one only with the user's okay, and the same cleanliness holds for it; if a change must touch any other repo or a git submodule, ask the user first.
- Record `git rev-parse --short HEAD` as Baseline commit in PROJECT.md, and each linked repo's as `- Linked repos: <dir> <commit>, ...` (or none). `exp.sh` builds every experiment on them; the baseline commit is where experiments start, and the baseline only if the user chose no other source.
- Freeze the reference outputs first: `verify --freeze` writes them to lx-work/baseline/reference/ from the project's own code; if the user chose another source, adapt the tool to freeze from it. Over 50 MB they go to lx-work/local/; commit their checksum and the command that regenerates them.
- Then run `verify --self-test` on the unmodified code. Against its own reference it shows that the outputs repeat in a fresh process and that verify can fail; against another source it gives the baseline commit's own difference, which the user sees.
- Bench the unmodified code: its numbers are the first best, and the baseline speed unless the user chose another source. Then measure the noise floor and the time breakdown as `rules/benchmarking.md` describes. In MAP.md, `- Measured on:` reads exactly `baseline commit <hash>`; the measurement conditions go in the paragraph below it.
- When the baseline speed comes from another environment, bench the same commit there, three runs, the way best is benched.
- Save the numbers in lx-work/baseline/numbers/, one directory per source: `<baseline commit hash>/` for the unmodified code (bench, A/A, time breakdown, verify), and the other environment's name for its bench JSON files. Save an environment snapshot (`pip freeze` or `conda list --export`) of each in lx-work/baseline/. Set `BASELINE` in the adapter to the baseline's bench JSON files, so every bench prints the baseline beside its own numbers.
- Fill STATUS.md: the baseline column from the baseline's files (name the environment when it is not this one) and best from the baseline commit's bench. Set the phase to `optimizing`, and commit lx-work.
- From here on the reference outputs are frozen.
