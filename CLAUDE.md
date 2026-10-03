---
tier: always
read_when: every session in a project that has lx-e2espeed/ (the SessionStart hook injects it)
---
# lx-e2espeed

Rules for making a trained model faster end to end, together with the user.
This is a collaboration, not an autonomous loop: you and the user discuss, and the user decides. Answer a question in words; reading files, code, logs, and history to answer it is fine.
Act only after the user explicitly says to: changing any file, running anything beyond read-only inspection, or installing. A go-ahead covers exactly what was agreed; anything more needs its own. If it is unclear whether the user asks or tells you to act, ask.
Exceptions: lx-work records (docs, indexes, notes, knowledge/) and their commits, written as things happen and listed at the end of your reply (PROJECT.md and DECISIONS.md still hold only what the user agreed); and a night run the user starts (`rules/night.md`), where nothing lands.
With this repo itself as the working directory, you are maintaining these rules: see `rules/writing.md`.

## Layout
- `<project>/`: the user's company repo; experiments change only its working tree, through `exp.sh`.
- `<project>/lx-e2espeed/`: these rules and the playbook, cloned from the main copy on this machine.
- `<project>/lx-work/`: everything about this project, in its own git repo, laid out as `rules/naming.md` shows.
Both lx directories are listed in the project's `.git/info/exclude`.

## Session start
If this file reached you without the hook's phase and status lines, the hooks are not installed: tell the user and propose `lx-e2espeed/setup.sh global`.
1. Read the rules doc the hook names for the current phase, and `git -C lx-work log --oneline -5` for recent work.
2. Tell the user in a few lines where things stand and what "Next" in STATUS.md proposes; start once they say so.

## Hard rules
- Do not change model weights or architecture. Outputs must meet the accuracy standard in `lx-work/PROJECT.md`.
- Do not edit reference outputs, tolerances, or verify criteria to make a change pass. Only the user changes them, recorded in `lx-work/DECISIONS.md`.
- Report results with the bench and verify tools' own summary lines, never a paraphrase or a claim without numbers.
- One landed change per experiment, so every gain or regression has one cause.
- Changing numeric precision, enabling `torch.compile`, and adding a dependency each need the user's explicit approval.
- Production code fails fast: never add a silent fallback to a slower path. Keep the reference path as an explicit offline configuration.
- Nothing about the project leaves lx-work. Cards and rule changes are abstract and reviewed by the user before they reach the main copy.
- Delivered code imports nothing from lx-work, lx-e2espeed, or sibling repositories; do not modify a sibling repository unless the user says so.
- In the project repo, do not commit, stash, reset, clean, or switch branches unless the user asks; commits use the user's own git identity with no co-author lines. Change its code only inside an experiment or night trial, ended by `exp.sh keep`, `park`, or `drop`, never by hand, so between them the tree equals the baseline plus the kept patches.
- Run every job that loads a model or data under a hard memory cap, at most half the RAM and sized from its estimated or slice-measured peak: `systemd-run --scope --quiet -p MemoryMax=<cap> -p MemoryLimit=<cap> <command>` (commands in these rules omit it). An uncapped job once froze a no-swap server for two days. Light commands such as setup.sh and the checkers take a small cap, for example 2G: a memory guard may block any uncapped Python or script.

## Defaults
- Every file in both lx repos is in English and written for an AI (`rules/writing.md`); anything for people goes in your reply.
- Send long command output to `lx-work/local/logs/` and read only the summary; raw output floods the context.
- When the user corrects how you work, record it in `lx-work/FEEDBACK.md`.
- Change lx files with Edit or Write, or with shell commands that name the lx directory, so the checker hook runs.
- When compacting, keep the current experiment, the latest numbers, and pending user decisions.

## Working loop
onboarding (ends with the baseline) → repeat (profile → propose → user picks → plan → user approves → candidates → verify → bench → user picks → land or drop → one lx-work commit) → wrap-up. A night run (`rules/night.md`) only parks trials for the user.

## Doc map
Read only what the current step needs.

Always: this file and `lx-work/STATUS.md`.

Once, when entering the named phase:
- `rules/onboarding.md`: first sessions in a new project.
- `hardware/rtx-5090.md`: during onboarding.
- `rules/wrap-up.md`: when the user says the project or a milestone is done.
- `rules/night.md`: when the user starts a night run, and the first session after one.

On demand:
- `rules/experiment.md`: phase optimizing; proposing, running, or reviewing an experiment.
- `rules/benchmarking.md`: before an A/B, judging a difference level, or changing a bench or verify tool.
- `rules/naming.md`: the lx-work layout and its sync after a pull; naming files, directories, branches, and commits.
- `rules/writing.md`: writing or editing any doc in either lx repo, or changing anything in lx-e2espeed.
- `playbook/INDEX.md`: looking for ideas after profiling; open only the matching cards.
- Skills: ours in `skills/`, NVIDIA's in `vendor/skills/`; `rules/experiment.md` says which to load. These rules win over a skill; skill output goes to `lx-work/local/`.
- `templates/`: lx-work docs, notes, cards, and the generic tools.
- lx-work: `PROJECT.md` agreed facts, `DECISIONS.md`, `FEEDBACK.md`, `BACKLOG.md` leads, `MAP.md` the timed path and its time shares, `tools/INDEX.md`, `exp/INDEX.md`, `night/INDEX.md`.
- `lx-work/knowledge/INDEX.md`: before reading project code to learn the model's purpose, callers, contract, or deployment.
