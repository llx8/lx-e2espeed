---
tier: on-demand
read_when: reviewing the rules, or asking why a rule is the way it is
---
# Changelog

Newest first. One entry per change to rules, templates, hooks, or the playbook structure, with the reason.

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

## 2026-09-24: bounded growth
- FEEDBACK.md is written without reading it (grep, then raise a count or append), capped at
  60 lines, and cleared of entries once they become rule changes. Without this, every
  correction would cost a full read and the file would grow for the whole project.
- One rule in writing.md now bounds every growing file: the experiment index archives,
  FEEDBACK.md clears resolved entries, DECISIONS.md drops superseded entries near its limit.
