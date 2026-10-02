---
tier: on-demand
read_when: creating files, directories, branches, or commit messages, deciding where something goes in lx-work, or after pulling lx-e2espeed
---
# Naming

Names are ASCII only. `hooks/check_docs.py` enforces the file and directory rules.

| What | Rule | Example |
|---|---|---|
| Directories, general files | lowercase kebab-case | `gen-inputs/`, `run.sh` |
| Python files and packages | lowercase snake_case, so they can be imported | `load_model.py` |
| Fixed docs | uppercase fixed names, one word | `STATUS.md`, `NOTE.md`, `INDEX.md`, `FEEDBACK.md` |
| Experiment and night trial directories | three-digit number plus 1 to 4 words; numbers are never reused, counted separately in exp/ and night/ | `003-cuda-graph-decode` |
| Tool directories | named by purpose; the entry file is `main.*` | `bench/main.py` |
| Cards | `playbook/<category>/<technique>.md` | `host/cuda-graph.md` |
| Dates | ISO 8601 | `2026-09-23` |

Never use `new`, `final`, `old`, `v2`, `tmp2`, `test1`, or similar. Git keeps versions; names describe content.

## lx-work layout
`setup.sh project` creates the docs, the indexes, knowledge/, and local/ from `templates/work-skeleton/`, and writes the clone's commit into `- Rules synced:` in STATUS.md. The rest is built where noted.

    lx-work/
    ├── STATUS.md      handoff: phase, numbers, in progress, next, waiting for the user
    ├── PROJECT.md     agreed facts: scope, workload, metric, accuracy, done-when, environment
    ├── DECISIONS.md   every decision with its date and reason
    ├── FEEDBACK.md    each correction the user made to how the work is done
    ├── BACKLOG.md     leads and known issues, each with evidence and where it went
    ├── MAP.md         end-to-end path as path:function pointers, with each step's share of time
    ├── knowledge/     what reading and the user taught: callers, contract, deployment; INDEX.md lists the files
    ├── adapter/       onboarding: load the model, build inputs, run once end to end
    ├── tools/         one directory per tool with entry file main.*; tools/INDEX.md lists them
    ├── baseline/      onboarding: numbers, frozen reference outputs, environment snapshot
    ├── exp/           INDEX.md results table, plus NNN-name/ per experiment from exp.sh:
    │                  NOTE.md, candidate/, change.patch, run.sh, result.json
    ├── night/         INDEX.md with the queue, plus NNN-name/ per night trial, laid out as in exp/
    └── local/         not in git: logs/, profiles/, output/, replay/, scratch/, large files

## Sync after a pull
After every pull of lx-e2espeed, bring lx-work up to the new skeleton; otherwise the checker and the next reader miss the fields and headings the rules now expect.
- `git -C lx-e2espeed diff --relative=templates/work-skeleton <Rules synced> HEAD -- templates/work-skeleton > lx-work/local/scratch/skeleton.patch`. If it is empty, skip to the last item.
- `git -C lx-work apply --reject local/scratch/skeleton.patch`. A hunk on a line the project already filled in lands in a `.rej` file: apply it by hand, keep the project's values, and delete the `.rej` file.
- Set `- Rules synced:` to `git -C lx-e2espeed rev-parse --short HEAD` and commit lx-work: `docs: align lx-work with rules <commit>: <what changed>`.

## Commit messages
- lx-work, one commit per experiment, kept or dropped: `exp-003: cuda graph for decode (kept, latency -18%)`, `exp-004: pinned host memory (dropped, inside noise)`. Night trials: `night-004: cuda graph for decode (parked, -18%)`. Other commits: `baseline: ...`, `tools: ...`, `docs: ...`.
- lx-e2espeed: `rules: ...`, `card(host): add cuda-graph`, `templates: ...`, `hooks: ...`.
- Commits delivered to the project follow the project's own conventions.
