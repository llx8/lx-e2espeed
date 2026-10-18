---
tier: on-demand
read_when: creating files, directories, branches, or commit messages, deciding where something goes in lx-work, or when the SessionStart hook says the lx-work skeleton changed
---
# Naming

Names are ASCII only. `hooks/check_docs.py` enforces the file and directory rules.

| What | Rule | Example |
|---|---|---|
| Directories, general files | lowercase kebab-case | `gen-inputs/`, `run.sh` |
| Python files and packages | lowercase snake_case, so they can be imported | `load_model.py` |
| Fixed docs | uppercase fixed names, one word | `STATUS.md`, `NOTE.md`, `INDEX.md`, `FEEDBACK.md` |
| Experiment and trial directories | three-digit number plus 1 to 4 words; numbers are never reused, counted separately in exp/ and trials/ | `003-cuda-graph-decode` |
| Tool directories | named by purpose; the entry file is `main.*` | `bench/main.py` |
| Scenarios and candidate variants | snake_case, since they are adapter keys and file names; a scenario named after its inputs, a variant after what it changes, never a role such as `main` or `test`: every report shows these names, and a role tells the user nothing (the user, 2026-10-11) | `sample200_mixed`, `one_image_one_face`; `gpu_decode_cpu_warp` |
| Cards | `playbook/<category>/<technique>.md` | `host/cuda-graph.md` |
| Dates | ISO 8601 | `2026-09-23` |

Never use `new`, `final`, `old`, `v2`, `tmp2`, `test1`, or similar. Git keeps versions; names describe content.

## lx-work layout
`setup.sh project` creates the docs, the indexes, knowledge/, and local/ from `templates/work-skeleton/`, and writes lx-e2espeed's commit into `- Rules synced:` in STATUS.md. The rest is built where noted.

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
    ├── baseline/      onboarding: numbers/<source>/ (baseline commit hash, other environment), frozen reference outputs, environment snapshots
    ├── exp/           INDEX.md results table, plus NNN-name/ per experiment from exp.sh:
    │                  NOTE.md, candidate/, ab-<variant>.json, change.patch, change.<dir>.patch per
    │                  changed linked repo (PROJECT.md), run.sh, result.json
    ├── trials/        INDEX.md with the queue, plus NNN-name/ per trial of a self-iteration run, laid out as in exp/
    └── local/         not in git: logs/, profiles/, output/, replay/, scratch/, large files

## Skeleton sync
When templates/work-skeleton changed after the `- Rules synced:` commit (the SessionStart hook says so), bring lx-work up to the new skeleton; otherwise the checker and the next reader miss the fields and headings the rules now expect.
- `git -C lx-e2espeed diff --relative=templates/work-skeleton <Rules synced> HEAD -- templates/work-skeleton > lx-work/local/scratch/skeleton.patch`. If it is empty, skip to the last item.
- `git -C lx-work apply --reject local/scratch/skeleton.patch`. A hunk on a line the project already filled in lands in a `.rej` file: apply it by hand, keep the project's values, and delete the `.rej` file.
- Set `- Rules synced:` to `git -C lx-e2espeed rev-parse --short HEAD` and commit lx-work: `docs: align lx-work with rules <commit>: <what changed>`.

## Commit messages
- lx-work, the closing commit of each experiment, kept or dropped: `exp-003: cuda graph for decode (kept, latency -18%)`, `exp-004: pinned host memory (dropped, inside noise)`; commits while it runs: `exp-003: plan (running)`. Trials: `trial-004: cuda graph for decode (parked, -18%)`. Other commits: `baseline: ...`, `tools: ...`, `docs: ...`.
- lx-e2espeed: `rules: ...`, `card(host): add cuda-graph`, `templates: ...`, `hooks: ...`.
- Commits delivered to the project follow the project's own conventions.

## Branch
- Before any change to a project repo or a linked repo, create `lx-dev` from its checked-out branch and switch to it: experiments, their commits, and the baseline commit all live there. The user's standing request (2026-10-18), so a hand-over covers it; an existing `lx-dev`, or the user's own uncommitted changes, go to the user first.
- Record `- Working branch:` in PROJECT.md: `lx-dev`, and the branch and commit each repo's `lx-dev` came from.
- `lx-dev` merges back into the branch it came from only when the work is done, with the user's go-ahead (`rules/wrap-up.md`); a dependency file then pins the merge, not `lx-dev`.
