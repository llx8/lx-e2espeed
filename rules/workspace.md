---
tier: once
read_when: a session starts in the workspace, the directory that holds lx-e2espeed/ above the projects (the SessionStart hook injects it)
---
# Working from the workspace

The workspace is the directory that holds these rules as `lx-e2espeed/` above the projects (for example a symlink to the shared rules directory). A session started there sees each project `setup.sh project` registered below it as one line from its STATUS.md, and works on one project at a time, so it reads only what that project needs.
A session started in a project's own root still works as before: the hook then injects that project alone.

## Start
- Tell the user in a few lines where each project stands, from the hook's lines, and ask which one to work on unless they already named it. Read nothing inside a project before that.
- A project, new or listed in phase onboarding but not as discussed, that the user hands over to optimize, in any words, starts the standard pass at once (`rules/standard-pass.md`): run `setup.sh project` in its root if it has no lx-work, then go on without a confirmation question.
- Once the user names one, `cd` into its root and work from there: every `lx-work/` path in the rules, and `exp.sh`, mean that project. Read `lx-e2espeed/CLAUDE.md` in full if this session has not yet, then follow its Session start for that project: its STATUS.md, the rules doc of its phase (the hook's line names it), and `git -C lx-work log --oneline -5`.
- A project's own CLAUDE.md, if it has one, loads once you read its files: follow it as in a session started there.

## One project at a time
- Read another project's lx-work or code only when the user asks, for example the project that pins the active one; say so in your reply.
- `lx-e2espeed/local/site.md` and the records it names are facts shared by the company's projects: reading them is not reading another project.
- Before switching, commit the active project's lx-work records, then start the next one as above. When compacting, keep which project is active and its root.
- Write what you learn about a project, and the user's corrections, into its lx-work (knowledge/, DECISIONS.md, FEEDBACK.md), never into Claude's memory: the workspace's memory loads in every session here, whichever project it works on.
- Only `setup.sh project`, run from a project's root, creates lx-work; never create one in the workspace itself.
- A self-iteration run (`rules/self-iteration.md`) covers the one project the user names.

## The shared GPU
Every project measures on the same GPU, and a session started in a project's own root may be running. Before a bench, A/B, or profile, check `nvidia-smi` for other processes (`rules/benchmarking.md`, GPU state) and tell the user instead of measuring next to one.
