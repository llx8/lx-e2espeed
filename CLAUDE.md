---
tier: always
read_when: every session in a project that has lx-work/ (the SessionStart hook injects it)
---
# lx-e2espeed

Rules for making a trained model faster end to end, together with the user.
This is a collaboration, not an autonomous loop: you and the user discuss, and the user decides. Answer a question in words, and only what it asks; reading to answer it is fine.
Act only after the user explicitly says to: changing any file, running anything beyond read-only inspection, or installing. A go-ahead covers exactly what was agreed; anything more needs its own. If unsure whether the user asks or tells you to act, ask.
Exceptions: lx-work records (docs, indexes, notes, knowledge/) and their commits, written as things happen and listed at the end of your reply (PROJECT.md and DECISIONS.md still hold only what the user agreed); and a self-iteration run the user starts (`rules/self-iteration.md`), where nothing lands.
With this repo as the working directory, you maintain these rules: see `rules/writing.md`.

## Layout
- `<project>/`: the user's company repo; experiments change only its tree and its linked repos', through `exp.sh`.
- `lx-e2espeed/`: these rules and the playbook, one repo for all projects; the SessionStart hook says where.
- `<project>/lx-work/`: all about this project, its own git repo, laid out as `rules/naming.md` shows.
- Workspace: the directory with `lx-e2espeed/` above the projects; sessions there follow `rules/workspace.md`.

## Session start
If no hook output came before this file, the hooks are not installed: propose `lx-e2espeed/setup.sh global` to the user.
1. Read the rules doc the hook names for the current phase, and `git -C lx-work log --oneline -5` for recent work.
2. Tell the user in a few lines where things stand and what "Next" in STATUS.md proposes; start once they say so.

## Hard rules
- Do not change model weights or architecture. Outputs must meet the accuracy standard in `lx-work/PROJECT.md`. Only research trials may differ, where PROJECT.md allows (`rules/self-iteration.md`).
- Do not edit reference outputs, tolerances, or verify criteria to make a change pass. Only the user changes them, recorded in `lx-work/DECISIONS.md`.
- Report results with the bench and verify tools' own summary lines, never a paraphrase or a number-free claim. In replies and tables, name each version, step, scenario, and level by what it is or does, never by a number or experiment ID: the user knows none.
- One landed change per experiment, so every gain or regression has one cause.
- Changing numeric precision, enabling `torch.compile`, and adding a dependency each need the user's explicit approval.
- Production code fails fast: never add a silent fallback to a slower path. Keep the reference path as an explicit offline configuration.
- Nothing about the project leaves lx-work: cards and rule changes stay abstract, checked before they are committed.
- Delivered code imports nothing from lx-work, lx-e2espeed, or sibling repos; leave sibling repos unchanged unless the user says so.
- In the project repo, do not commit, stash, reset, clean, or switch branches unless the user asks; commits use the user's own git identity with no co-author lines. Change its code only inside an experiment or trial, ended by `exp.sh keep`, `park`, or `drop`, so between them the tree equals the baseline commit plus the kept patches.
- Run every job that loads a model or data under a hard memory cap, at most half the RAM, sized from its estimated or slice-measured peak: `systemd-run --scope --quiet -p MemoryMax=<cap> -p MemoryLimit=<cap> <command>` (omitted in these rules). An uncapped job once froze a no-swap server for two days; light commands take a 2G cap, or a memory guard may block them.

## Defaults
- Every file in both lx repos is in English and written for an AI (`rules/writing.md`); anything for people goes in your reply.
- Send long output to `lx-work/local/logs/` and read only the summary.
- When the user corrects how you work, record it in `lx-work/FEEDBACK.md`.
- Change lx files with Edit, Write, or shell commands naming the lx directory, so the checker hook runs.
- When compacting, keep the project, its experiment, latest numbers, and pending user decisions.

## Working loop
onboarding (ends with the baseline) → repeat (profile → propose → user picks → plan → user approves → candidates → verify → bench → user picks → land or drop → closing lx-work commit) → wrap-up. A self-iteration run only parks trials for the user.

## Doc map
Read only what the current step needs.

Always: this file and `lx-work/STATUS.md`.

Once, when entering the named phase:
- `rules/onboarding.md`: first sessions in a new project.
- `hardware/rtx-5090.md` or `hardware/l20.md`: during onboarding, for the production GPU.
- `rules/wrap-up.md`: when the user says the project or a milestone is done.
- `rules/self-iteration.md`: when the user starts a self-iteration run, and the session after it.

On demand:
- `rules/experiment.md`: phase optimizing; proposing, running, or reviewing an experiment.
- `rules/benchmarking.md`: before an A/B, judging a difference level, or changing a bench or verify tool.
- `rules/reporting.md`: before reporting results or explaining work.
- `rules/naming.md`: the lx-work layout, its skeleton sync, and naming anything.
- `rules/writing.md`: writing any doc in either lx repo, or changing lx-e2espeed.
- `playbook/INDEX.md`: looking for ideas after profiling; open only the matching cards.
- Skills (ours in `skills/`, NVIDIA's in `vendor/skills/`): `rules/experiment.md` says which to load; these rules win; output goes to `lx-work/local/`.
- `templates/`: lx-work docs, notes, cards, and the generic tools.
- `lx-work/knowledge/INDEX.md`: before reading project code to learn its purpose, callers, contract, or deployment.
