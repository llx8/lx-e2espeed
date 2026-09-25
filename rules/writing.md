---
tier: on-demand
read_when: writing or editing any doc in lx-e2espeed or lx-work
---
# Writing docs

Applies to every doc in both lx repos, except the files in `vendor/`, which stay as upstream wrote them. Items marked (checked) are enforced by `hooks/check_docs.py`.

## Reading tiers
Every doc starts with frontmatter (checked):

    ---
    tier: always | once | on-demand
    read_when: <the situation that calls for reading it>
    ---

- always: read every session. Keep these short; every line costs context in every session.
- once: read when entering a phase. Its output lands in lx-work, so later sessions read the output instead.
- on-demand: read only when the current step needs it.

## Line limits (checked)
| Doc | Max lines |
|---|---|
| lx-e2espeed/CLAUDE.md and CHANGELOG.md | 100 |
| lx-work/STATUS.md | 60 |
| lx-work/FEEDBACK.md | 60 |
| experiment or night trial NOTE.md | 50 |
| playbook card | 40 |
| any other doc | 150 |
| exp/ARCHIVE.md | none: grep it, never read it whole |

At the limit, split or cut. Raise a limit only with the user, here and in the checker together.
Scripts have no line limit: a script is split by purpose, never by length.

## Structure
- References go one level deep: CLAUDE.md names every rules and hardware doc directly (checked). A doc may point to another, but must not depend on a chain of them.
- A doc over 100 lines starts with a `## Contents` list (checked).
- A doc made from a template keeps the template's `##` headings and `- Field:` lines, and CLAUDE.md keeps its sections (checked). Losing one breaks the next reader or script.
- Indexes and logs hold one line per entry, and files that grow during a project stay bounded. exp/INDEX.md moves old dropped rows to `ARCHIVE.md` next to it; FEEDBACK.md deletes entries once they became rule changes; DECISIONS.md deletes superseded entries and CHANGELOG.md its oldest entries when they near their limit. Git history keeps everything removed.
- Point, do not copy: refer to code as `path:function`. Pasted code goes stale.
- Re-measuring the same thing updates its existing record instead of adding a new section. Raw data lives in one artifact file; the doc keeps the adopted result with its measurement conditions.
- Each fact lives in one place and other docs point to it. Long lines repeated across docs fail the check (checked).

## Style
- Write for an AI reader: no person reads these files. Skip introductions, walkthroughs, and anything that does not change what the reader does.
- English only, scripts and comments included (checked). The project's own code in an experiment's change.patch and candidate/ keeps its language.
- Give the reason next to any rule whose reason is not obvious. A reader that knows why applies it with better judgment.
- Hard rules only where a mistake is costly or hard to undo; everything else is guidance.
- Keep version-dependent facts out of rules, or mark them with the date they were checked.
- Use one term per concept, as defined below.

## Glossary
- end to end: the timed path agreed in PROJECT.md.
- baseline: the unmodified project code, measured during onboarding.
- best: the baseline plus all kept patches, which is the project tree between experiments.
- candidate: code in an experiment's candidate/ directory that runs beside the project code; landing applies the chosen variant to the project code.
- difference, difference level (L0 to L3), operator accuracy, cold start: defined in `rules/benchmarking.md`.
- experiment: one change with one hypothesis, recorded in exp/ with its change.patch.
- night trial: an experiment of a night run, in night/. It never stays in the tree: it is parked (patch saved and reversed) or dropped, and the user may adopt a parked one into an experiment.
- delivered: a kept experiment whose patch the user committed to the project; the baseline commit includes it.
- kept patches: the change.patch files of kept experiments. Applied to the baseline commit in number order, they rebuild the project tree.
- verify: compare outputs with the baseline reference outputs.
- bench: measure the metric.
- noise floor: the run-to-run spread of the bench on unchanged code; smaller differences are not gains.
- time breakdown: each end-to-end step's share of the total time, recorded in MAP.md.
- plan: the written change, verification, and expected gain in an experiment's NOTE.md, before any code changes.
- card: one playbook entry describing one technique.
- main copy: the lx-e2espeed repo on this machine that every project clone comes from; each clone's `origin`. Its own `origin` is the cloud repo, pushed only at the end of a project or when the user asks.
