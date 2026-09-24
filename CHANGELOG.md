---
tier: on-demand
read_when: reviewing the rules, or asking why a rule is the way it is
---
# Changelog

Newest first. One entry per change to rules, templates, hooks, or the playbook structure, with the reason.

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

## 2026-09-24: workflow upgrade
- Experiments follow find → propose → plan → change: the user reviews the proposal and, for
  larger changes, a written plan in NOTE.md. Review is worth most before code exists
  (HumanLayer, research-plan-implement; Claude Code best practices, explore-plan-code).
- Baseline adds a noise floor (repeated runs on unchanged code) and a time breakdown per
  end-to-end step. Gains inside the noise do not count; the breakdown bounds what a change can
  win (Amdahl's law) and points to where to look (Brendan Gregg, anti-methodologies).
- exp/INDEX.md became a results table; keep decisions weigh complexity against gain
  (Karpathy, autoresearch). Long command output goes to lx-work/local/logs/.
- Onboarding also settles done-when, target hardware, the Python environment, and profiler
  commands. What to do when the company code moves is covered in experiment.md.
- New lx-work/FEEDBACK.md captures the user's corrections when they happen, so the rules
  review at wrap-up works from records instead of memory.
- The SessionStart hook names the rules doc for the current phase and re-injects the
  handoff after compaction. The checker now requires NOTE.md and INDEX.md to agree, kept
  experiments to carry change.patch, run.sh, and result.json, and STATUS.md to be no older
  than the newest experiment. PROJECT.md records the rules version a project started with.
- CLAUDE.md now holds the full lx-work layout and which step builds each part. It was spread
  over several rules, so no single place described it.
- Renamed templates/lx-work to templates/work-skeleton: the old name read like the real lx-work.
- Sync through a local main copy: every project clones from it, and changes move back by
  pulling into it (wrap-up step 5). Version control stays on this machine with no hosting;
  pulling is used because git refuses pushes into a repo with a checked-out branch.
