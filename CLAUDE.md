---
tier: always
read_when: every session in a project that has lx-work/ (the SessionStart hook injects it)
---
# lx-e2espeed

Rules for making a trained model faster end to end, together with the user.
A collaboration: you and the user discuss, and the user decides, outside the runs below. Answer a question in words, only what it asks; reading to answer it is fine.
Act only after the user explicitly says to: changing any file, running anything beyond read-only inspection, or installing. A go-ahead covers exactly what was agreed; anything more needs its own. If unsure whether the user asks or tells you to act, ask.
Exceptions: lx-work records and their commits, written as they happen, listed at the end of your reply (PROJECT.md and DECISIONS.md hold only what the user agreed, or a pass's marked defaults); and the runs the user starts, a hand-over included (`rules/self-iteration.md`, `rules/standard-pass.md`).
With this repo as the working directory, you maintain these rules: see `rules/writing.md`.

## Layout
- `<project>/`: the user's company repo; experiments change only its tree and its linked repos', through `exp.sh`.
- `lx-e2espeed/`: these rules and the playbook, one repo for all projects.
- `<project>/lx-work/`: all about this project, its own git repo, laid out as `rules/naming.md` shows.
- Workspace: the directory above the projects holding `lx-e2espeed/`; see `rules/workspace.md`.

## Session start
No hook output before this file means the hooks are not installed: propose `lx-e2espeed/setup.sh global`.
1. Read the rules doc the hook names for the current phase, and `git -C lx-work log --oneline -5`.
2. A new project (no lx-work, or phase onboarding) handed over to optimize, in any words, starts the standard pass at once: read `rules/standard-pass.md`, ask nothing first (missed 10-12 and 10-16). Only "look/discuss first" or "just read" means the discussed onboarding.
   Otherwise tell the user in a few lines where things stand and what "Next" in STATUS.md proposes; start once they say so.

## Hard rules
- Do not change model weights or architecture. Outputs must meet the accuracy standard in `lx-work/PROJECT.md`. Only research trials may differ, where PROJECT.md allows (`rules/self-iteration.md`).
- Do not edit reference outputs, tolerances, or verify criteria to make a change pass. Only the user changes them, recorded in `lx-work/DECISIONS.md`.
- Report results with the bench and verify tools' own summary lines, never a paraphrase or a number-free claim. In replies and tables, name each version, step, scenario, and level by what it is or does, never by a number or experiment ID.
- One landed change per experiment, so every gain or regression has one cause.
- Changing numeric precision, enabling `torch.compile`, and adding a dependency each need the user's explicit approval. A production image's package at its version is alignment (`rules/onboarding.md`).
- Production code fails fast: never add a silent fallback to a slower path. Keep the reference path as an explicit offline configuration.
- Nothing about the project leaves lx-work: cards and rule changes stay abstract, checked before they are committed.
- Delivered code imports nothing from lx-work, lx-e2espeed, or sibling repos; leave sibling repos unchanged unless the user says so.
- In the project repo, do not commit, stash, reset, clean, or switch branches unless the user asks; commits use the user's own git identity with no co-author lines. Change its code only inside an experiment or trial, ended by `exp.sh keep`, `park`, or `drop` (`rules/experiment.md`).
- Run every job that loads a model or data under a hard memory cap (`systemd-run --scope --quiet -p MemoryMax=<cap> -p MemoryLimit=<cap>`, omitted in these rules), at most half the RAM, sized from its estimated or measured peak; light commands take 2G. An uncapped job once froze a server for days.

## Defaults
- Both lx repos: English, written for an AI (`rules/writing.md`); anything for people goes in your reply.
- Send long output to `lx-work/local/logs/` and read the summary.
- When the user corrects how you work, record it in `lx-work/FEEDBACK.md`.
- Change lx files with Edit, Write, or shell commands naming the lx directory, so the checker runs.
- When compacting, keep the project, its experiment, latest numbers, and pending decisions.

## Working loop
onboarding or a standard pass → repeat (profile → propose → user picks → plan → approve → candidates → verify → bench → user picks → land or drop → commit) → wrap-up. A self-iteration run only parks trials; a pass lands identical outputs, parks the rest.

## Doc map
Read only what the step needs. Always: this file and `lx-work/STATUS.md`.

Once, when entering the named phase:
- `rules/onboarding.md`: first sessions in a new project.
- `hardware/rtx-5090.md` or `hardware/l20.md`: during onboarding, for the production GPU.
- `rules/wrap-up.md`: when the user says the project or a milestone is done.
- `rules/self-iteration.md`: when the user starts one, and the session after it.
- `rules/standard-pass.md`: at a hand-over or start, during a pass, and the session after it.

On demand:
- `rules/experiment.md`: phase optimizing; proposing, running, or reviewing an experiment.
- `rules/benchmarking.md`: before an A/B, a difference level, or changing a bench or verify tool.
- `rules/reporting.md`: before reporting results or explaining work.
- `rules/delivery.md`: at wrap-up steps 1 and 2.
- `rules/naming.md`: the lx-work layout, its skeleton sync, and naming anything.
- `rules/writing.md`: writing any doc in either lx repo, or changing lx-e2espeed.
- `playbook/INDEX.md`: looking for ideas after profiling; open only matching cards.
- Skills (NVIDIA's, `vendor/skills/`): `rules/experiment.md` says which to load; these rules win; output to `lx-work/local/`.
- `templates/`: lx-work docs, cards, and the generic tools.
- `lx-work/knowledge/INDEX.md`: before reading project code to learn what it is for.
