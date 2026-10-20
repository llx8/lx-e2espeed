---
tier: on-demand
read_when: writing or editing any doc in lx-e2espeed or lx-work, or changing anything in lx-e2espeed
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

## Size limits (checked)
| Doc | Max lines |
|---|---|
| lx-e2espeed/CLAUDE.md and CHANGELOG.md | 100 |
| lx-work/STATUS.md | 60 |
| lx-work/FEEDBACK.md | 60 |
| experiment or trial NOTE.md | 50 |
| playbook card | 40 |
| any other doc | 150 |
| exp/ and trials/ ARCHIVE.md | none: grep it, never read it whole |

Bytes, too (checked): CLAUDE.md at most 6,000 and STATUS.md at most 3,000, because the SessionStart hook injects both, and Claude Code cuts hook output above about 10 KB to a 2 KB preview; every other lx-e2espeed doc at most 12,000, except CHANGELOG.md; and one line of prose outside code fences at most 500 bytes.
A bullet at the line limit becomes sub-bullets, or its detail moves to a card or a tool's doc.
What goes where: a rule (what to do, when, what needs the user) in rules/; a technique (how to make something faster, with its evidence) in a card; how a tool is built in `templates/tools/DESIGN.md` or the tool itself; why a rule exists, with dates and the user's words, in CHANGELOG.md, leaving at most a short clause in the rule. A rule the checker enforces is stated once, where it is acted on, and not repeated elsewhere.

At the limit, split or cut. Raise a limit only with the user, here and in the checker together.
Scripts have no line limit: a script is split by purpose, never by length.

## Structure
- References go one level deep: CLAUDE.md names every rules and hardware doc directly (checked). A doc may point to another, but must not depend on a chain of them.
- A doc over 100 lines starts with a `## Contents` list (checked).
- A doc made from a template keeps the template's `##` headings and `- Field:` lines, and CLAUDE.md keeps its sections (checked). Losing one breaks the next reader or script.
- Indexes and logs hold one line per entry, and files that grow during a project stay bounded. exp/INDEX.md moves old dropped rows, and trials/INDEX.md old dropped or adopted rows, to `ARCHIVE.md` next to it; FEEDBACK.md deletes entries once they became rule changes; DECISIONS.md deletes superseded entries and CHANGELOG.md its oldest entries when they near their limit. Git history keeps everything removed.
- Point, do not copy: refer to code as `path:function`. Pasted code goes stale.
- Re-measuring the same thing updates its existing record instead of adding a new section. Raw data lives in one artifact file; the doc keeps the adopted result with its measurement conditions.
- Each fact lives in one place and other docs point to it. Long lines repeated across docs fail the check (checked).

## Style
- Write for an AI reader: no person reads these files. Skip introductions, walkthroughs, and anything that does not change what the reader does.
- English only, scripts and comments included (checked). The project's own code in an experiment's change.patch and candidate/ keeps its language; non-English model data (inputs, reference outputs) goes to `lx-work/local/`.
- Give the reason next to any rule whose reason is not obvious. A reader that knows why applies it with better judgment.
- Hard rules only where a mistake is costly or hard to undo; everything else is guidance.
- Keep version-dependent facts out of rules, or mark them with the date they were checked.
- Use one term per concept, as defined below.

## Maintaining lx-e2espeed
- Give every new check a probe in `hooks/test_checks.py`, which proves each check can fail, and every change to `exp.sh` a step in `hooks/test_exp.py`, which runs it on a throwaway project; the pre-commit hook runs both.
- Log every change to rules, templates, or hooks in `CHANGELOG.md` with its reason.
- During project work, change rules or cards at wrap-up, or earlier when the user asks; either way only on the user's go-ahead.
- lx-e2espeed is one working tree that the sessions of every project edit, and a saved change reaches every project at once. Before editing, `git -C lx-e2espeed status --porcelain` must print nothing; anything else is another session's unfinished change, so stop and ask the user. Try a change to a script or hook on a copy first.
- The user does not review the diff: the AI maintains lx-e2espeed. Check it yourself before committing: nothing project-specific, the checker and `hooks/test_checks.py` pass, CHANGELOG.md logs it. Then commit at once, staging your files by name: `git add -A` or `commit -a` would take in another session's unfinished edit.
- Never rewrite its history (no amend, rebase, or reset): lx-work names rules commits in `- Rules synced:`, and a rewritten commit is gone. Pushing to the cloud follows `rules/wrap-up.md` step 5.

## Glossary
- end to end: the timed path agreed in PROJECT.md.
- baseline: what every result is compared with, a speed and outputs (the reference outputs), set during onboarding. By default both come from the unmodified code at the baseline commit; the user may choose another source for either, recorded in DECISIONS.md: for example the framework model at full precision when the project's own outputs are known to be wrong, or production's speed in its previous environment.
  "Baseline" in reports, STATUS.md, and every "vs baseline" column means this; `BASELINE` in the adapter names its bench files.
- quality baseline: the network in FP32 by its framework inside the unmodified request path, with its seed spread; how good outputs are is measured against it, while the baseline's reference outputs only show whether outputs changed (`rules/quality.md`).
- baseline commit: the project commit every experiment starts from (PROJECT.md, `- Baseline commit:`); exp.sh rebuilds the project tree from it. Its own bench is the first best, and the baseline speed only when the user chose no other source. Delivering kept patches or moving to newer company code changes the baseline commit, never the baseline.
- best: the baseline commit plus all kept patches, which is the project tree between experiments. In a report it is "before", the A/B's best side.
- candidate: code in an experiment's candidate/ directory that runs beside the project code; landing applies the chosen variant to the project code. In a report it is "after".
- difference, difference level (L0 to L3), operator accuracy, cold start: defined in `rules/benchmarking.md`.
- experiment: one change with one hypothesis, recorded in exp/ with its change.patch.
- trial: an experiment of a self-iteration run (`rules/self-iteration.md`), in trials/. It never stays in the tree: it is parked (patch saved and reversed) or dropped, and the user may adopt a parked one into an experiment.
- delivered: a kept experiment whose patch the user committed to the project; the baseline commit includes it.
- kept patches: the change.patch files of kept experiments, with their change.<dir>.patch for each linked repo (PROJECT.md, `- Linked repos:`). Applied in number order to the baseline commit, and to each linked repo's commit, they rebuild the project tree and the linked repos.
- verify: compare outputs with the frozen reference outputs.
- bench: measure the metric.
- noise floor: the run-to-run spread of the bench on unchanged code; smaller differences are not gains.
- time breakdown: each end-to-end step's share of the total time, recorded in MAP.md.
- plan: the written change, verification, and expected gain in an experiment's NOTE.md, before any code changes.
- card: one playbook entry describing one technique.
