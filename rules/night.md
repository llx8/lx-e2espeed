---
tier: once
read_when: the user starts a night run, and the first session after one
---
# Night run

The user starts a night run before leaving. You try methods alone, one night trial per method in lx-work/night/NNN-name/, until no idea is left; the next session reports them, and the user picks what lands.
It is the one autonomous mode, and it is safe because nothing lands: every trial ends with its change saved as a patch and reversed.

## Before the user leaves
- Phase optimizing, no running experiment, `exp.sh check` passes, verify and bench pass, and the noise floor is current. This tree, the baseline plus the kept patches, is the base of every trial tonight.
- Agree on the end time (default: when the user is back) and the scope. Default scope: any method inside the hard rules in CLAUDE.md. Numeric precision and `torch.compile`, which need approval by day, may be tried because nothing lands; flag them for the user. Nothing is installed or downloaded, and nothing outside the project tree and lx-work changes: a method that needs a dependency is queued as blocked.
- A permission prompt stops the night: confirm with the user that the session runs without them.
- Write the queue in night/INDEX.md: every method with evidence that it applies here, from a fresh profile, MAP.md, BACKLOG.md, the playbook, and the skill routing table in rules/experiment.md, ordered by expected gain (bounded by its step's share). Leave out directions DECISIONS.md rules out for this project. Put the night run under "In progress" in STATUS.md, and commit lx-work.

## Each trial
1. `exp.sh try <1 to 4 words>` checks the tree and creates night/NNN-name/ with NOTE.md and a running row.
2. Write Evidence, Hypothesis, and Plan. Build candidates in its candidate/ and compare them as in rules/experiment.md step 4: verify, and the A/B against the untouched tree, at least three rounds.
3. A gain beyond the noise floor, or less code at equal speed, that passes verify: land the smallest change in the project tree, verify the real code, save run.sh (the exact commands that reproduce the trial's numbers from its base, writing its output next to itself, found from its own path, since adopt copies it into exp/) and result.json, then `exp.sh park`. It saves change.patch, checks that the patch rebuilds the tree, reverses it, and checks that the tree is back at the base.
4. No gain, a failed verify, or a dead end: `exp.sh drop`, and record what was learned.
5. A difference level beyond the accepted one in PROJECT.md is not a failure: the user may accept it. Land and park it, and say so under "Needs the user". Where the method forks on a user judgment, give each side its own trial.
6. Fill in NOTE.md and the row, update the queue, and commit lx-work: `night-004: cuda graph for decode (parked, -18%)`.

Timebox each trial (default 90 minutes). Past it, drop the trial with what was learned, and queue the rest as a day task. Kill any run that takes twice its usual time.

## All night
- Never end your turn to ask: write the question under "Needs the user" and go on with the next method. Stop only when the queue is empty and a fresh look (profile, playbook, skills) adds nothing, at the end time, or when the environment breaks (the same infrastructure error twice).
- Trials do not build on each other: each starts from the base, so each patch applies alone. A method that only makes sense on top of another is queued as a day task.
- The GPU runs nothing else, and clocks are locked as in rules/benchmarking.md.
- Rules, cards, references, and tolerances stay untouched. A problem with the rules goes to FEEDBACK.md for the next rules review.

## Ending the night
`exp.sh check` passes: the tree is at the base. In STATUS.md, "In progress" is none, and "Waiting for the user" says `night run <date>: N parked, M dropped, K queued; report them (rules/night.md)`. Commit lx-work.

## The morning report
- In your reply, not in a file. One table of parked trials, best first: the method, the A/B per scenario against the base, the difference level, verify, what it touches and its risk, and what the user must decide. Then one line each for dropped trials and for methods still queued, with the reason.
- Say whether the base moved since the night; if it did, the numbers are against an older tree. Trials do not add up: the user picks one at a time, and each later pick is measured on top of the earlier ones.
- `exp.sh adopt NNN` turns a parked trial into the next experiment, carrying NOTE.md and candidate/. On an unchanged base it applies the patch and copies run.sh and result.json: run verify once, then `exp.sh keep`. On a moved base it applies nothing: A/B the candidate against the current tree first (rebuild a candidate that copies code a kept experiment changed since), then land as in rules/experiment.md step 5.
- Trials the user passes on stay parked; they are still under "Next" at wrap-up. `exp.sh export night/NNN` rebuilds any trial's code from its own base.
