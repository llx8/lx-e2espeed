---
tier: once
read_when: first sessions in a new project (STATUS.md phase is onboarding)
---
# Onboarding a project

Goal: agree with the user on what "faster" means here, then measure the unmodified project precisely. The user's review matters most here: a misunderstanding misdirects every later experiment.
Which mode: CLAUDE.md, Session start. A hand-over runs `rules/standard-pass.md`: step 1 below, then its defaults instead of step 2; one naming reference projects ports their changes too (environment move, then optimizations) as pass items. The Mode line keeps a later "go on" in a discussed onboarding from starting the pass.

Discussed onboarding: two steps, each started once the user says so. Copy this line into your reply and tick it off:

    - [ ] 1 Set up and read  - [ ] 2 Discuss, then the baseline

## 1. Set up and read
Setup:
- Not a git repo: ask the user (a hand-over covers it), `git init` and commit the code as delivered; nothing is pushed.
- Create `lx-dev` in each repo (`rules/naming.md`).
- Turn MPS on in the project's Dockerfile (`playbook/serving/mps-multi-worker.md`), the first change (`exp.sh`, kept at any speed); after onboarding the checker fails a project without it.
- `lx-e2espeed/setup.sh project`, from the project root on a hand-over or the user's word: excludes lx-work/ from the repo, creates it from `templates/work-skeleton/` as a git repo, and registers the project for the checker's private terms and the workspace list. The rest of lx-work follows `rules/naming.md`.

Read the project:
- Read `lx-e2espeed/local/site.md` first when it exists: this machine's facts shared across the company's projects (images, CI, service framework, tested libraries, reference code), each pointing to its record. It is never committed.
- Trace the end-to-end path (input → preprocessing → host-to-device copy → model → device-to-host copy → postprocessing), skipping unrelated code, into MAP.md as `path:function` pointers, with an ASCII flow for a multi-stage pipeline.
- Note per stage its device, dtype, layout, and shape, every device or representation change between stages, normal and maximum shapes, batch policy, and ROI or tile behavior. Likely bottlenecks stay hypotheses until measured.
- Find how it runs: the Python environment (called by its absolute path; `conda run` or a polluted PATH can pick another Python), the entry command, and the weights. A weight file of a few hundred bytes is a Git LFS pointer: `git lfs pull` before measuring.
- For a diffusion pipeline, record the base model, the weights' quantization, pipeline and scheduler, LoRA formats and ranks, and the reference workflow (a ComfyUI graph), which is the specification; fusing a LoRA or changing its rank is a contract change.
- Code issues and leads go into BACKLOG.md with their evidence.
- Project, company, product, and model names go into Private terms in PROJECT.md as you meet them, so the checker keeps them out of lx-e2espeed; provisional until the user confirms them in step 2 ("Waiting for the user" in STATUS.md).
- The rest of what the reading teaches goes into `lx-work/knowledge/` as INDEX.md there says: callers and meaning (business.md), exact formats (contract.md), how production builds and runs it (deployment.md). Later sessions read these instead of the code.

Build the harness, so step 2 has numbers to look at (`rules/benchmarking.md`):
- Check the profilers (`nsys --version`, `ncu --version`, one `ncu --set full` run); both ship in the CUDA toolkit's bin/, maybe off PATH, and permissions can change after a container restart. Record the working commands in PROJECT.md, in backticks.
- If CUDA looks unavailable, check whether the command ran in a sandbox before concluding the GPU or driver is missing, and say which it was.
- Write `lx-work/tools/check-env/`: one command listing every gap at once (modules, CUDA in the framework, compiler, TensorRT headers matching the runtime, engines loadable, current GPU). Run it first whenever the environment misbehaves.
- Copy `templates/tools/adapter/` to `lx-work/adapter/` and `bench/`, `verify/`, `cpu-profile/` to `lx-work/tools/`, then fill every TODO the code answers (scenarios, stages, loading, one call, output arrays); the levels, time budget, and `BASELINE` wait for step 2. Copy `run-image/` when a number must come from a production image, `engine-noise/` when the model runs as a built engine (L1 is the spread of rebuilds), and `load-test/` at wrap-up.
- Bench the unmodified code once, a first number for the discussion, not yet the baseline.
- Change a generic tool only for a real project need; bring a general fix back to the templates at wrap-up (`templates/tools/DESIGN.md`). List every tool in tools/INDEX.md; before a tool grows past its question, tell the user what it adds and costs in lines.

## 2. Discuss, then the baseline
Explain your understanding first, the issues found in step 1 apart from the items to decide. Then one or two items at a time in plain words: what it means, its options, which you would pick and why; the user decides. Never the whole list at once, never a quick yes; items the code or `local/site.md` answers need only a confirmation.
Record the agreed parameters in PROJECT.md, each decision with its reason in DECISIONS.md, and what the user explains about callers, contract, or deployment in the knowledge files, where an answer replaces its open question. They belong to this project only.
- End-to-end scope: where timing starts and stops.
- Workload: input shapes, batch sizes, request pattern, warm or cold start.
- Metric: latency by default; throughput or memory only if the user says so. Every result is compared with the baseline (`rules/writing.md`, Glossary): the baseline commit's own bench unless the user names another source.
- Bench time budget for one run: it sets WARMUP and REPEAT in the adapter and goes under Metric.
- Accuracy standard (`rules/quality.md`): the identity reference, the quality baseline and its metric, the thresholds of L0 to L3 or none, and the accepted level.
- Contracts: the public interface downstream code relies on (argument order, return structure, channel order, sizes), the pre/post tensor contract (dtype, range, layout, color order, interpolation, rounding), and who consumes the output. Changing one needs the downstream impact stated first.
- Deployment: how production runs it (image and versions, how it loads the repo and resolves weights, workers per GPU, MPS), the maximum input, and what a delivery must include (CI steps that rebuild engines, dependency locks, runtime libraries). Align the local environment with production's versions where possible. A service repo pinning a separate algorithm repo ends its delivery with a merge and a new pin (`rules/wrap-up.md`).
- Done when: the target, diminishing returns (three experiments in a row inside the noise floor), or a deadline.
- Target hardware: if production is not this machine, gains here may not transfer; say so and agree how to confirm them. Read the production GPU's doc (`hardware/rtx-5090.md`, `hardware/l20.md`).
- Constraints: code that must not change, allowed dependencies, techniques ruled out (pinned memory), and the private terms from step 1, confirmed and completed.
- Environment: which Python environment the tools run in, by default the one `local/site.md` names. A package the project's code imports, or its CI or init scripts install, is added without asking, pinned to the target image's version (`local/site.md`); if pip would change an installed package's version, leave it to the user, since every project shares that environment. Any other package is a new dependency.

Once the user has confirmed, the baseline follows; it ends the step and the phase.
- Put the agreed levels and time budget into the adapter. The project tree must be clean (`git status --porcelain` prints nothing): the user commits or stashes their own changes; untracked environment files such as weights go into `.git/info/exclude` with their okay.
  `exp.sh` sees this repo and its linked repos: git checkouts at the project top that the project ignores (a sibling algorithm repo or its symlink), which experiments may change. Link one only with the user's okay, equally clean; a change to any other repo or a submodule needs the user first.
- Record `git rev-parse --short HEAD` as Baseline commit in PROJECT.md, and each linked repo's as `- Linked repos: <dir> <commit>, ...` (or none).
- Freeze the references first, the quality baseline with its seed spread too (`rules/quality.md`): `verify --freeze` writes them to lx-work/baseline/reference/ (over 50 MB to lx-work/local/, with their checksum and regenerating command committed); another source chosen by the user needs the tool adapted.
- `verify --self-test` on the unmodified code: against its own reference it shows outputs repeat in a fresh process and verify can fail; against another source it gives the baseline commit's own difference, which the user sees.
- Bench the unmodified code: its numbers are the first best, and the baseline speed unless the user chose another source (then bench the same commit there, three runs). Measure the noise floor and the time breakdown (`rules/benchmarking.md`). In MAP.md, `- Measured on:` reads exactly `baseline commit <hash>`, the conditions in the paragraph below it.
- Save the numbers in lx-work/baseline/numbers/, one directory per source (`<baseline commit hash>/`, or the other environment's name), and an environment snapshot (`pip freeze` or `conda list --export`) of each in lx-work/baseline/. Set `BASELINE` in the adapter to the baseline's bench JSON files.
- Fill STATUS.md (the baseline column, naming its environment when it is not this one, and best), set the phase to `optimizing`, and commit lx-work. From here on the reference outputs are frozen.
