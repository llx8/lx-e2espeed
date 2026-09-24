---
tier: on-demand
read_when: creating files, directories, branches, or commit messages
---
# Naming

Names are ASCII only. `hooks/check_docs.py` enforces the file and directory rules.

| What | Rule | Example |
|---|---|---|
| Directories, general files | lowercase kebab-case | `gen-inputs/`, `run.sh` |
| Python files and packages | lowercase snake_case, so they can be imported | `load_model.py` |
| Fixed docs | uppercase fixed names, one word | `STATUS.md`, `NOTE.md`, `INDEX.md`, `FEEDBACK.md` |
| Experiment directories | three-digit number plus 1 to 4 words; numbers are never reused | `003-cuda-graph-decode` |
| Tool directories | named by purpose; the entry file is `main.*` | `bench/main.py` |
| Cards | `playbook/<category>/<technique>.md` | `host/cuda-graph.md` |
| Dates | ISO 8601 | `2026-09-23` |

Never use `new`, `final`, `old`, `v2`, `tmp2`, `test1`, or similar. Git keeps versions; names describe content.

## Commit messages
- lx-work, one commit per experiment, kept or dropped: `exp-003: cuda graph for decode (kept, latency -18%)`, `exp-004: pinned host memory (dropped, inside noise)`. Other commits: `baseline: ...`, `tools: ...`, `docs: ...`.
- lx-e2espeed: `rules: ...`, `card(host): add cuda-graph`, `templates: ...`, `hooks: ...`.
- Commits delivered to the project follow the project's own conventions.
