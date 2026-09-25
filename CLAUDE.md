---
tier: always
read_when: every session in a project that has lx-e2espeed/ (the SessionStart hook injects it)
---
# lx-e2espeed

Rules for making a trained model faster end to end, together with the user.
This is a collaboration, not an autonomous loop: discuss before acting and let the user decide. The one exception is a night run the user starts (`rules/night.md`), where nothing lands.
With this repo itself as the working directory, you are maintaining these rules: see "Maintaining this repo".

## Layout
- `<project>/`: the user's company repo. Experiments change only its working tree, through `exp.sh`; it gets no branch and no commit from you.
- `<project>/lx-e2espeed/`: these rules and the playbook, cloned from the main copy on this machine (the clone's `origin`).
- `<project>/lx-work/`: everything about this project, in its own git repo.
Both lx directories are listed in the project's `.git/info/exclude`.

## lx-work layout
`setup.sh project` creates the docs, both indexes, and local/ from `templates/work-skeleton/`. The rest is built where noted.

    lx-work/
    ├── STATUS.md      handoff: phase, numbers, in progress, next, waiting for the user
    ├── PROJECT.md     agreed facts: scope, workload, metric, noise floor, accuracy, done-when, environment
    ├── DECISIONS.md   every decision with its date and reason
    ├── FEEDBACK.md    each correction the user made to how the work is done
    ├── BACKLOG.md     leads and known issues, each with evidence and where it went
    ├── MAP.md         end-to-end path as path:function pointers, with each step's share of time
    ├── adapter/       onboarding: load the model, build inputs, run once end to end
    ├── tools/         one directory per tool with entry file main.*; tools/INDEX.md lists them
    ├── baseline/      onboarding: numbers, frozen reference outputs, environment snapshot
    ├── exp/           INDEX.md results table, plus NNN-name/ per experiment from exp.sh:
    │                  NOTE.md, candidate/, change.patch, run.sh, result.json
    ├── night/         INDEX.md with the queue, plus NNN-name/ per night trial, laid out as in exp/
    └── local/         not in git: logs/, profiles/, output/, replay/, scratch/, large files

## Session start
If this file reached you without the SessionStart hook, the hooks and skills are not installed on this machine: ask the user, then run `lx-e2espeed/setup.sh global`.
1. The hook injects `lx-work/STATUS.md` and names the rules doc for the current phase; read that doc.
2. Run `git -C lx-work log --oneline -5` to see recent work.
3. Before the first change of the session, run verify and bench once. A broken environment should be caught before it is blamed on a change.
4. Tell the user in a few lines where things stand, then continue from "Next" in STATUS.md.

## Hard rules
Each one guards against a mistake that is costly or hard to undo.
- Do not change model weights or architecture. Outputs must match the baseline within the tolerance in `lx-work/PROJECT.md`.
- Do not edit reference outputs, tolerances, or verify criteria to make a change pass. Only the user changes them, recorded in `lx-work/DECISIONS.md`.
- Report results with the bench and verify tools' own summary lines, never a paraphrase or a claim without numbers.
- One landed change per experiment, so every gain or regression has one cause.
- Ask before changing numeric precision, enabling `torch.compile`, installing or adding a dependency, or doing anything the user has not agreed to.
- Production code fails fast: never add a silent fallback to a slower path. Keep the reference path as an explicit offline configuration.
- Nothing about the project leaves lx-work. Cards and rule changes are abstract and reviewed by the user before they reach the main copy.
- Delivered code imports nothing from lx-work, lx-e2espeed, or sibling repositories; do not modify a sibling repository unless the user says so.
- In the project repo, do not commit, stash, reset, clean, or switch branches unless the user asks; commits use the user's own git identity with no co-author lines. Change its code only inside an experiment or night trial, and end each with `exp.sh keep`, `park` (night trials), or `drop`, never by hand, so between them the tree always equals the baseline plus the kept patches.

## Defaults
- Metric: latency, unless PROJECT.md says otherwise. Accuracy tolerance, dependencies, and done-when are agreed per project.
- Readers: every file in both lx repos is written for an AI, in English, scripts and comments included. Text the model needs in another language (inputs, reference outputs) stays in `lx-work/local/` or in the project's own files; project code in `change.patch` and candidate copies keeps its language. Anything for people goes in your reply.
- Send long command output (profiling, builds, long runs) to `lx-work/local/logs/` and read only the summary; raw output floods the context.
- When the user corrects how you work, record it in `lx-work/FEEDBACK.md` right away: grep for a similar line and raise its count, otherwise append one line.
- Change lx files with Edit or Write, or with shell commands that name the lx directory, so the checker hook runs.
- When compacting, keep the current experiment, the latest numbers, and pending user decisions.

## Working loop
onboarding → baseline → repeat (profile → propose → user picks → plan → candidates → verify → bench → land or drop → one lx-work commit) → wrap-up.
Night run: queue → repeat (try → candidates → verify → A/B → land and park, or drop → one lx-work commit) → morning report → user picks → `exp.sh adopt`.

## Doc map
Read only what the current step needs. Each doc states its tier in its frontmatter.

Always: this file and `lx-work/STATUS.md`.

Once, when entering the named phase:
- `rules/onboarding.md`: first sessions in a new project.
- `hardware/rtx-5090.md`: during onboarding; copy the facts that matter into PROJECT.md.
- `rules/benchmarking.md`: before writing a bench or verify tool.
- `rules/wrap-up.md`: when the user says the project or a milestone is done.
- `rules/night.md`: when the user starts a night run, and the first session after one.

On demand:
- `rules/experiment.md`: phase optimizing; proposing, running, or reviewing an experiment.
- `rules/naming.md`: creating files, directories, branches, or commit messages.
- `rules/writing.md`: writing or editing any doc in either lx repo.
- `playbook/INDEX.md`: looking for ideas after profiling; open only the matching cards.
- Skills: ours in `skills/`, NVIDIA's in `vendor/skills/`, installed by `setup.sh global`; `rules/experiment.md` says which to load. These rules win over a skill; skill output goes to `lx-work/local/`.
- `templates/`: lx-work docs, experiment notes, cards, and the generic tools with their adapter (`templates/tools/`).
- lx-work: `PROJECT.md`, `DECISIONS.md`, `FEEDBACK.md`, `BACKLOG.md`, `MAP.md`, `tools/INDEX.md`, `exp/INDEX.md`, `night/INDEX.md`.

## Maintaining this repo
- Follow `rules/writing.md`; `hooks/check_docs.py` enforces its checked items.
- Log every change to rules, templates, or hooks in `CHANGELOG.md` with its reason.
- During project work, change rules or cards only at wrap-up, after the user's review. Push the main copy to its cloud `origin` only at the end of a project (wrap-up step 5) or when the user asks.
