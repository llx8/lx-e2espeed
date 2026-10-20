---
tier: always
read_when: every session in a project that has lx-work/ (the SessionStart hook injects it)
---
# lx-e2espeed

Rules for making a trained model faster end to end, together with the user.
A collaboration: you and the user discuss, and the user decides, except in the runs below. Answer a question in words, only what it asks; reading to answer it is fine.
Act only after the user explicitly says to: changing any file, running anything beyond read-only inspection, or installing. A go-ahead covers exactly what was agreed; anything more needs its own. If unsure whether the user asks or tells you to act, ask.
Exceptions: lx-work records and their commits, written as they happen, listed at the end of your reply (PROJECT.md and DECISIONS.md hold only what the user agreed, or a pass's marked defaults); and the runs the user starts, a hand-over included (`rules/self-iteration.md`, `rules/standard-pass.md`).
In this repo, you maintain these rules (`rules/writing.md`).

## Layout
- `<project>/`: the user's company repo; experiments change only its tree and its linked repos', through `exp.sh`.
- `lx-e2espeed/`: these rules and the playbook, one repo for all projects.
- `<project>/lx-work/`: this project's records, its own git repo (`rules/naming.md`).
- Workspace: the directory above the projects holding `lx-e2espeed/`; see `rules/workspace.md`.

## Session start
No hook output before this file means the hooks are not installed: propose `lx-e2espeed/setup.sh global`.
1. Read the rules doc the hook names for the current phase, and `git -C lx-work log --oneline -5`.
2. A new project handed over to optimize, in any words, starts the standard pass at once: read `rules/standard-pass.md`, ask nothing first. New: no lx-work, or onboarding the hook calls not discussed.
   "Look/discuss first" or "just read" means the discussed onboarding: add `- Mode: discussed` to STATUS.
   Otherwise tell the user in a few lines where things stand and what "Next" in STATUS.md proposes; start once they say so.

## Hard rules
- Do not change model weights or architecture. Outputs must meet the accuracy standard in `lx-work/PROJECT.md`. Only research trials may differ, where PROJECT.md allows (`rules/self-iteration.md`).
- Do not edit reference outputs, tolerances, or verify criteria to make a change pass. Only the user changes them, recorded in `lx-work/DECISIONS.md`.
- Report results with the bench and verify tools' own summary lines, never a paraphrase. In replies and tables, name each version, step, scenario, and level by what it is or does, never by a number or experiment ID.
- One landed change per experiment, so every gain or regression has one cause.
- Changing numeric precision, enabling `torch.compile`, and adding a dependency each need the user's explicit approval. Environment alignment is not one (`rules/onboarding.md`).
- Production code fails fast: never add a silent fallback to a slower path. Keep the reference path as an explicit offline configuration.
- Nothing about the project leaves lx-work: cards and rule changes stay abstract, checked before they are committed.
- Delivered code imports nothing from lx-work, lx-e2espeed, or sibling repos; leave sibling repos unchanged unless the user says so.
- In the project repo, do not commit, stash, reset, clean, or switch branches unless the user asks; commits use the user's own git identity with no co-author lines. Change its code only inside an experiment or trial, ended by `exp.sh keep`, `park`, or `drop` (`rules/experiment.md`).
- Run every job that loads a model or data under a hard memory cap (`systemd-run --scope --quiet -p MemoryMax=<cap> -p MemoryLimit=<cap>`, omitted in these rules), at most half the RAM, sized from its estimated or measured peak; light commands take 2G. An uncapped job once froze a server for days.
- Every job that touches a GPU, here or on L20, goes through the queue, so sessions never measure on top of each other: `gpulane run --exclusive --mem <cap> -- CMD` for a bench, A/B, or noise floor; `--shared` for verify, export, or a build; `--host L20` there. `--mem` is the memory cap. A job longer than a few minutes: `gpulane submit ...`, then `gpulane wait <id>`; an A/B of several runs holds `gpulane lease <minutes>`. `gpulane ls` shows the queue; never sleep to wait for the GPU.

## Defaults
- Both lx repos: English, written for an AI (`rules/writing.md`); anything for people goes in your reply.
- Send long output to `lx-work/local/logs/` and read the summary.
- When the user corrects how you work, record it in `lx-work/FEEDBACK.md`.
- Change lx files with Edit, Write, or shell commands naming the lx directory, so the checker runs.
- When compacting, keep the project, its experiment, numbers, and pending decisions.

## Working loop
onboarding or a standard pass → repeat (profile → propose → user picks → plan → approve → candidates → verify → bench → user picks → land or drop → commit) → wrap-up. A self-iteration run only parks trials; a pass lands identical outputs, parks the rest.

## Doc map
Read only what the step needs. Always: this file and `lx-work/STATUS.md`.

- `rules/INDEX.md`: every rules and hardware doc, with when to read it. The hook names the phase's first doc.
- `rules/reporting.md`: before reporting results, any table of numbers, or explaining work; its table forms are fixed.
- `playbook/INDEX.md`: what to try after profiling, ranked by past projects; open only matching cards.
- Skills (`vendor/skills/`): `rules/experiment.md` says which; these rules win; output to `lx-work/local/`.
- `templates/`: lx-work docs, cards, generic tools.
- `lx-work/knowledge/INDEX.md`: before reading project code to learn its purpose, callers, contract, or deployment.
