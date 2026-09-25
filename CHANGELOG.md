---
tier: on-demand
read_when: reviewing the rules, or asking why a rule is the way it is
---
# Changelog

Newest first. One entry per change to rules, templates, hooks, or the playbook structure, with the reason.

## 2026-09-25: no line limit for scripts
- The user lifted the script limit: some scripts are long by nature, and splitting exp.sh would spread shared functions across files. Docs keep their limits, which keep them maintainable.

## 2026-09-25: night runs
- Night run (rules/night.md, lx-work/night/): the user asked for a mode that tries every method with evidence overnight, each in its own directory, reproducible and applicable to the project at any time, for the user to pick from in the morning. It is the one autonomous mode and lands nothing: `exp.sh try` starts a trial on the current tree, `park` saves its verified patch and reverses it, and `adopt` turns a parked trial into an experiment, applying the patch when the base has not moved. `export night/NNN` rebuilds a trial from its own base. The SessionStart hook names any experiment or trial left running.
- `exp.sh delivered`: after the user committed the kept patches, `exp.sh` refused every command, because HEAD was no longer the baseline. It now checks that HEAD is exactly the baseline plus the kept patches, makes HEAD the new baseline, and marks those experiments delivered. `export NNN` rebuilds from the item's own Base line, so a delivery does not change it.
- The skeleton PROJECT.md kept the `<rules-version>` token that setup.sh fills; the previous rename had broken it.
- From the first real night run: the queue leaves out directions DECISIONS.md rules out (a trial of a ruled-out precision change would have been queued), less code at equal speed may be parked, and run.sh writes next to itself (a copied run.sh wrote into the trial's directory after adopt). The adapter docstring no longer ties candidates to exp/.

## 2026-09-25: review after the pilots
- The bench A/A mode is built in (`--ab baseline`); a pilot had to write its own control candidate. The noise floor is per scenario (`adapter.NOISE_MS` is a dict), as the rules already required; an A/B without one exits. Under three rounds the A/B gives no verdict: an identical-code A/B with one round read as a 38 ms loss. Rounds swap which side runs first.
- The checker flags template placeholders left in STATUS.md or PROJECT.md after onboarding; a pilot finished with one in STATUS.md. "Rules version" became "Rules version at onboarding": later pulls do not update it.
- The verify and cpu-profile templates flush stderr before redirecting it, as the bench does.

## 2026-09-25: pilot, second project
- Generic bench, verify, and cpu-profile in templates/tools/, with an adapter skeleton that holds everything project-specific. The second project would otherwise rewrite the first project's tools and could repeat its measurement mistakes.
- Onboarding checks that weights are real files, not Git LFS pointers. The verify template refuses to overwrite frozen references: once optimized code lands, freezing again would hide every change.

## 2026-09-24: pilot on a real project
A pilot re-optimized an image CNN service from its unoptimized commit (6 experiments, 5 kept, -58% at L0) to find what the rules got wrong.
- Onboarding asks a few plain questions with proposed answers; the MAP.md template has a place for the stage flow; the adapter loads candidates too.
- Benchmarking: inline steps split with a function-level profiler; native logs redirected at the file-descriptor level; an A/B mode with one fresh process per scenario, since scenarios sharing a process shifted each other by several ms and produced a fake gain; noise floor per scenario from A/A runs, re-measured after large gains; targeted tests for branches the scenarios never reach; A/B medians are the recorded numbers.
- Templates: per-scenario numbers with a main scenario, no `<metric>` placeholders, lx-work ignores Python caches (8 .pyc files had been committed).
- Checker: project code in change.patch and candidate/ keeps its language (exp.sh keep produced a patch the English check rejected); finished notes cannot keep placeholders; template-shape errors also cover fields added by newer rules.
- Wrap-up commits kept patches from a temporary index; the rules repo refuses rule changes without a CHANGELOG entry, because the pilot's own rule fixes went unlogged.

## 2026-09-24: the user's GPU inference skill and two past projects
- Split the user's GPU inference optimization skill (its commit db32202) into eight skills in skills/: profiling,
  dataflow, memory, CNN TensorRT, torch.compile, diffusion, MPS hang, and delivery. Translated
  to English, project and company names removed. Its fixed defaults (FP16 tensor contract,
  weight store, Nunchaku, no pinned memory) became onboarding questions: the user said every
  project needs its own analysis. setup.sh global installs them with NVIDIA's.
- Experiments explore with candidates in exp/NNN/candidate/ beside untouched project code, as
  the user worked before, then land the chosen variant through exp.sh keep.
- Benchmarking follows the user's measurement habits: cold start apart, P99 and peak memory,
  locked clocks for A/B, re-baseline after hardware changes, real inputs only, difference
  levels L0 to L3, "difference" kept apart from "operator accuracy", checks that can fail.
- Onboarding settles contracts, deployment, maximum input, and difference levels; PROJECT.md
  has sections for them, and BACKLOG.md holds leads. New hard rules: fail fast without silent
  fallbacks, no imports from lx or sibling repos, torch.compile only with approval.
- Sixteen playbook cards with measured evidence from the two projects.
- The edit hook shows at most 15 errors and no warnings, to save context.
- Always-loaded text kept short: one-line skill descriptions (quoted, checked), a few words per
  playbook entry, a shorter CLAUDE.md; wrap-up points to the delivery skill instead of copying it.

## 2026-09-24: NVIDIA performance skills
- Copied five skills unchanged from NVIDIA TensorRT-LLM (Apache-2.0) into vendor/skills/:
  nsys, ncu, CUDA graphs, host-device syncs, timing code. They are generic PyTorch and CUDA
  guidance written for coding agents.
- setup.sh global installs them to ~/.claude/skills/, because Claude Code skips skills inside
  git-ignored directories, and every project ignores lx-e2espeed/. The SessionStart hook warns
  when the installed hooks or skills differ from the clone.
- Left out their autopilot and TensorRT-LLM-specific skills; UPSTREAM.md says why and how to update.
- The checker skips vendor/; these rules win where a skill differs.

## 2026-09-24: experiments without a branch
- The project repo gets no branch and no commit. Each experiment's change is saved as a patch in
  lx-work, and exp.sh keeps the tree equal to the baseline commit plus the kept patches: keep
  checks the patches rebuild the tree, drop reverses the change and checks the tree is back.
  Undoing by hand could wipe the user's own changes, and git diff missed new files.
- exp.sh also does the bookkeeping (number, NOTE.md, index row), and refuses to start when
  HEAD moved or the tree drifted. Every experiment ends in one lx-work commit, kept or
  dropped; lx-work uses one branch, main.
- The checker now requires docs made from templates to keep their headings and fields,
  CLAUDE.md to keep its sections, and the hook's phase docs to exist. It checks every text
  file for English, and keeps the private terms in PROJECT.md out of lx-e2espeed.
- The edit hook also checks the lx repo the shell is in, not only ones a command names.
- Wrap-up says how to sync when another project reached the main copy first.
- exp.sh warns about ignored files an experiment wrote, since no patch holds them. A commit-msg
  hook keeps private terms out of lx-e2espeed commit messages. Onboarding covers a project
  delivered without git, and submodules. setup.sh also excludes Python caches.
- A failed technique is recorded as the condition it failed under, not as a verdict: it did not
  fit one project, and the next may differ. Cards count successes and failures.

## 2026-09-24: files for AI only
- Every file in both lx repos is written for an AI; no person reads them. The user hears
  results in the chat, so a human guide or report in the repos only costs upkeep.
- README.md is gone. Machine setup moved into CLAUDE.md, project setup into onboarding step 1.
  The SessionStart hook now also fires in a project that has lx-e2espeed/ but no lx-work/ yet,
  so onboarding starts without a person following a guide.
- No report file: STATUS.md, exp/INDEX.md, and DECISIONS.md already hold the record, and
  wrap-up tells the user the outcome in the reply.
- Every doc needs frontmatter, CHANGELOG.md included, and CHANGELOG.md is bounded like the rest.
  lx-work scripts are checked for English too.
