---
tier: once
read_when: the user starts a self-iteration run, and the first session after one
---
# Self-iteration run

The user starts a self-iteration run and then leaves or watches; it runs by day or by night (the user renamed it from "night run" on 2026-10-11, same procedure). You try methods alone, one trial per method in lx-work/trials/NNN-name/, until no idea is left; then you report them, and the user picks what lands.
It is one of the two autonomous modes (the other, the standard pass in `rules/standard-pass.md`, lands identical-output gains), and it is safe because nothing lands: every trial ends with its change saved as a patch and reversed.

## Before it starts
- Phase optimizing, no running experiment, `exp.sh check` passes, verify and bench pass, and the noise floor is current. This tree, best (the baseline commit plus the kept patches), is the base of every trial in the run.
- Agree on the end time (default: when the user is back) and the scope. Default scope: any method inside the hard rules in CLAUDE.md. Numeric precision and `torch.compile`, which need approval otherwise, may be tried because nothing lands; flag them for the user. Nothing is installed or downloaded, and nothing outside the project tree and lx-work changes: a method that needs a dependency is queued as blocked.
- A permission prompt stops a run the user is away from: confirm with the user that the session runs without them.
- Write the queue in trials/INDEX.md: every method with evidence that it applies here, from a fresh profile, MAP.md, BACKLOG.md, the playbook, and the skill routing table in rules/experiment.md, ordered by expected gain (bounded by its step's share). Leave out directions DECISIONS.md rules out for this project. Put the run under "In progress" in STATUS.md as `- Run: self-iteration run, session <id>` (the id the SessionStart hook printed), and commit lx-work.

## Each trial
1. `exp.sh try <1 to 4 words>` checks the tree and creates trials/NNN-name/ with NOTE.md and a running row.
2. Write Evidence, Hypothesis, and Plan. Build candidates in its candidate/ and compare them as in rules/experiment.md step 4: verify, and the A/B against the untouched tree, at least three rounds.
3. A gain beyond the noise floor, or less code at equal speed, that passes verify: land the smallest change in the project tree, verify the real code, save run.sh (the exact commands that reproduce the trial's numbers from its base, writing its output next to itself, found from its own path, since adopt copies it into exp/) and result.json, then `exp.sh park`. It saves change.patch, checks that the patch rebuilds the tree, reverses it, and checks that the tree is back at the base.
4. No gain, a failed verify, or a dead end: `exp.sh drop`, and record what was learned.
5. A difference level beyond the accepted one in PROJECT.md is not a failure: the user may accept it. Land and park it, and say so under "Needs the user". Where the method forks on a user judgment, give each side its own trial.
6. Fill in NOTE.md and the row, update the queue, and commit lx-work: `trial-004: cuda graph for decode (parked, -18%)`.

Timebox each trial (default 90 minutes). Past it, drop the trial with what was learned, and queue the rest as a day task.

## During the run
- Never end your turn to ask: write the question in the trial's "Needs the user" column, or under "Waiting for the user" in STATUS.md when it concerns no single trial, and go on with the next method. Stop only when the queue is empty, a fresh look (profile, playbook, skills) adds nothing, and every lead is closed by a measurement, not an estimate; at the end time; or when the environment breaks (the same infrastructure error twice).
- When the user watches, send the short status lines of rules/reporting.md; the run still never waits for an answer.
- While the Run line names this session, the Stop hook refuses the end of the turn (at most 20 times in a row while STATUS.md is unchanged), so the run cannot stop early. The ending steps replace the line; a session that resumes the run puts its own id there first.
- Trials do not build on each other: each starts from the base, so each patch applies alone. A method that only makes sense on top of another is queued as a day task.
- The GPU runs nothing else, and clocks are locked as in rules/benchmarking.md.
- Nobody may be there to reboot a frozen machine: every run stays under the memory cap in CLAUDE.md and is killed when MemAvailable in /proc/meminfo falls below 20% of RAM. Rerun a crashed run only once the crash is understood.
- Rules, cards, references, and tolerances stay untouched. A problem with the rules goes to FEEDBACK.md for the next rules review.

## Ending the run
`exp.sh check` passes: the tree is at the base. In STATUS.md, "In progress" is none (the Run line replaced), and "Waiting for the user" says `self-iteration run <date>: N parked, M dropped, K queued; report them (rules/self-iteration.md)`. Commit lx-work.

## The report
- In your reply when the run ends, or in the first session after it, not in a file. One table of parked trials, best first: the method, the A/B per scenario against the base, the difference level, verify, what it touches and its risk, and what the user must decide. Then one line each for dropped trials and for methods still queued, with the reason, the run's entries under "Waiting for the user", and the FEEDBACK.md lines the run added.
- Say whether the base moved since the run; if it did, the numbers are against an older tree. Trials do not add up: the user picks one at a time, and each later pick is measured on top of the earlier ones.
- The user telling you to use a parked method or library, in any words, is its adoption: adopt it, never report it as still waiting (the user, 2026-10-17).
- `exp.sh adopt NNN` turns a parked trial into the next experiment, carrying NOTE.md and candidate/. On an unchanged base it applies the patch and copies run.sh and result.json: run verify once, then `exp.sh keep`. On a moved base it applies nothing: A/B the candidate against the current tree first (rebuild a candidate that copies code a kept experiment changed since), then land as in rules/experiment.md step 5.
- Trials the user passes on stay parked; they are still under "Next" at wrap-up. `exp.sh export trials/NNN` rebuilds any trial's code from its own base.
- A run that ends with the queue empty, in a project that allows the research stage, says that the research stage is next.

## Research stage
Only where PROJECT.md has a `- Research stage:` line that allows it, in the user's words, and only once the user starts it; it is a self-iteration run of research trials, so everything above applies. Why: once engineering finds nothing more, a research project still needs a contribution of its own, so the user wants the method itself tried (the user, 2026-10-10 and 2026-10-12).
- A research trial changes the method, not its implementation: a formula from a paper, modified, or one of your own, for more speed, better quality, or both. In these trials, and only there, the hard rules on weights, architecture, and the accuracy standard give way as far as the PROJECT.md line says, and so does the approval for precision, since nothing lands.
  Every other rule holds: the memory cap, nothing installed or downloaded (queue it as blocked), one idea per trial from the same base, parked or dropped.
- Queue each with `research:` before the method in trials/INDEX.md. Its Evidence names the source (paper, year, section, equation) or the reasoning behind your own formula.
- Judge each on speed and quality together: the bench's summary line and the quality metric that the PROJECT.md line names, on the same inputs as the base. Report the difference level too, but a research trial changes outputs by design, so the level alone decides nothing.
- An adopted research trial lands as an experiment only once the user accepts its quality, recorded in DECISIONS.md with the numbers.
