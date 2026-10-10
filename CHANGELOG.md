---
tier: on-demand
read_when: reviewing the rules, or asking why a rule is the way it is
---
# Changelog

Newest first. One entry per change to rules, templates, hooks, or the playbook structure, with the reason.

## 2026-10-10: linked repos in exp.sh, snapshot keeps the index mtime
- A service project keeps its algorithm code in a separate repo, a gitignored symlink at the project top. exp.sh saw only the project repo, so it could not keep, park, or drop a change there, and exports lacked that code. PROJECT.md now names linked repos (`- Linked repos: <dir> <commit>, ...` or none). exp.sh checks, snapshots, and replays each like the project tree from its commit, saves change.<dir>.patch, applies or reverses all patches or none, and records the linked commits in the Base line; `export` writes each into `<name>/<dir>/`, and `delivered` moves their commits. Projects without the field and old Base lines work as before.
- exp.sh copied the index for its snapshot with a fresh mtime, which defeats git's racy-file check: a file changed to the same size within the second it was staged read as unchanged, and `check` failed on a correct tree. A scratch test of the linked repos hit it. The copy keeps the mtime now.
- Checker: change.<dir>.patch keeps the project's language like change.patch, and MAP.md's breakdown check reads it under `<dir>/`. The clean fixture holds one, and a probe covers the breakdown check.

## 2026-10-09: algorithm repo merge and new pin at delivery, MPS on by default
- The user, in a service project that pins its algorithm repo in a dependency file: at the end the algorithm's dev branch is merged back into the branch the work started from, or, when the algorithm repo has only its main branch, into a new develop created from it; the dependency file then pins that branch at its newest commit. Wrap-up step 1 says so, and onboarding step 3 points to it, so the layout is planned for it from the start.
- The user: MPS is on by default in production. Onboarding step 3 now assumes it unless the user says otherwise, so MPS hazards are checked from the first experiment instead of found at delivery.

## 2026-10-03: baseline versus baseline commit, three sets per result
- "Baseline" named two things: the glossary's unmodified code at the baseline commit, and what results are compared with. A user chose production's previous environment as the speed to compare with, and the AI then reported the baseline commit's bench in this environment as the baseline. The glossary now separates the baseline (speed and reference outputs that every result compares with, the baseline commit's own unless the user chose another source) from the baseline commit (where experiments start); best is "before" in a report and the candidate "after". Rules, templates, skills, exp.sh, and the checker use the two names; "speed reference" is gone.
- Formats: MAP.md reads `- Measured on: baseline commit <hash>`, NOTE.md `- Base: baseline commit <hash> + kept ...`. exp.sh still parses old Base lines; the checker requires the new MAP.md line, and a probe covers the old one. `exp.sh export baseline` became `export baseline-commit`.
- The user wants three sets for every optimization result, baseline, before, and after, each with median, P90, P99, and memory. The bench reads the baseline's own bench files from the new `adapter.BASELINE` and prints them beside every run; an A/B prints the three sets per scenario and writes every round of both sides to `--out`, where before they lived only in scratch files the next A/B overwrote. The exp-note and STATUS.md templates have the rows.
- Numbers from another session or environment are not a paired A/B: a change against the baseline is labeled descriptive, and a milestone or wrap-up benches both sides in one session, alternating three runs each.
- Onboarding step 5 keeps the baseline's numbers in baseline/numbers/<source>/ and sets `BASELINE`; when its speed comes from another environment, it benches the same commit there three times. `exp.sh delivered` moves the baseline commit; the baseline stays, and only the user changes it.
- Skills no longer say "baseline" for the eager path, an accuracy check, a warm memory level, or a reference kernel family.

## 2026-10-03: conflicts found at a project's baseline step
- Onboarding step 5 ran verify before saving the reference that verify reads, and never named `verify --freeze`. It now freezes first, then runs `verify --self-test` on the unmodified code and says what that run shows.
- The rules assumed the unmodified code supplies both the reference outputs and the speed reference, never subtracted numbers across environments, and required difference-level thresholds. A user chose otherwise: outputs from the framework model at full precision, speed from production's previous environment, no thresholds. The glossary, benchmarking.md, step 3, the PROJECT.md template, and the hard rule on outputs now allow each, as a user decision in DECISIONS.md.
- The checker flagged every `<...>` in PROJECT.md after onboarding, while step 4 records commands there whose metavariables look the same. It now skips code spans, fenced blocks, and indented lines, where templates put no fields; a probe shows a placeholder beside a code span is still caught.
- MAP.md's `- Measured on:` holds only the stamp, but nothing said where the measurement conditions go: the paragraph below it, in the template, step 5, and the checker's message.
- An A/B read `clocks.sm` to see whether the lock held; a GPU running below its lock under the power cap cannot show that. The trigger is now a reboot or driver reload since the lock.
- The working loop showed the baseline as a stage after onboarding; it is onboarding's last step. Step 5 no longer restates where the noise floor goes. Kept on purpose: the go-ahead and memory-cap rules in both CLAUDE.md and the user's global instructions (the rules must hold without the latter), and one-line reminders at the steps where they apply.

## 2026-10-01: Rules synced field, small caps for light commands
- Bringing an existing lx-work up to a changed skeleton had no recorded procedure, and which rules commit lx-work matched lived in free text in STATUS.md; the first two projects each did it twice by hand. STATUS.md now has `- Rules synced:`, which `setup.sh project` fills, and `rules/naming.md` gives the sync after a pull: apply the skeleton diff since that commit, merge rejected hunks by hand, update the field.
- A memory guard blocked `setup.sh global` and the checker for running without a cap. The cap rule now says light commands take a small one.

## 2026-10-01: hook output size, bench and verify fixes, one home per fact
- Claude Code cut the SessionStart output (10.8 KB) to a 2 KB preview, so the entry rules never reached the session. The hook now prints the short, urgent lines first (paths, what to read if cut, phase and its rules doc, warnings), then STATUS.md, then CLAUDE.md; the checker caps CLAUDE.md at 6,000 bytes and STATUS.md at 3,000. The lx-work layout moved to `rules/naming.md`, the maintaining rules to `rules/writing.md`.
- Bench: samples SM clock, temperature, and power during the run, since readings before and after miss throttling in between. The A/B sides are best (or the `--project` tree) and the candidate, and the A/A is `--ab best`: between experiments the project tree is best, not baseline. P99 needs 100 samples; below that the bench reports the maximum.
- Verify: the self-test also changes an EXACT array, skips scenarios that already fail, and fails when no scenario had a value to change; a project with only EXACT outputs proved nothing before.
- The profiling skill's benchmark and e2e scripts are gone: they duplicated the project bench without its rules (A/B, fresh processes, noise floor).
- Clock: locked at all times; the user allows a reset in a special case, with their okay, instead of never.
- One home per fact: the noise floor in `adapter.NOISE_MS` and STATUS.md (PROJECT.md dropped its line); the bench time budget is a step 3 item and a PROJECT.md field; knowledge files dropped their Read at lines, which the INDEX.md column holds; night/INDEX.md archives like exp/INDEX.md; night.md no longer repeats the kill rule from benchmarking.md; the cloud push rule lives only in wrap-up step 5.
- Onboarding step 2 fills the private terms provisionally as names come up, so the checker protects the rules repo before step 3 confirms them.

## 2026-10-01: work on the user's go-ahead, memory cap, locked clock, consistency review
- The user works with the AI as two people discussing: a question gets an answer in words, and nothing changes (files, runs beyond reading, installs) until the user says to. CLAUDE.md now says so first. Rules that acted on their own changed: session start no longer runs verify and bench (each experiment's verify and A/B already run the baseline beside the change, and check-env exists for a misbehaving environment); every plan waits for approval, even a one-sentence one; onboarding steps start on the user's word, and step 3 explains each item's options and the AI's pick instead of asking for a quick yes.
- lx-work records (docs, indexes, notes, knowledge/) and their commits are the exception: the user does not want to be asked about them, only told, so the AI writes them as things happen, FEEDBACK.md included, and lists them at the end of its reply.
- The SM clock stays locked and is never reset: the user has the GPU to themselves, and a lock that stays keeps every session at the same clock. A reboot drops it, so an A/B reads the clock first.
- Memory cap: an uncapped job filled a no-swap server's RAM and froze it for two days. Every job that loads a model or data runs under a systemd-run memory cap of at most half the RAM; a night run also kills a run when free RAM falls below 20%, and reruns a crash only once it is understood.
- Consistency fixes from the first project on the maintained rules: rules may change mid-project when the user asks, as knowledge/ did; benchmarking.md is on demand, since every A/B and difference level needs it after onboarding; a milestone wrap-up skips the cloud push and sets the phase back to optimizing; `exp.sh check` fails while an item runs, because the tree is not at the base; a night question that concerns no single trial goes under "Waiting for the user" in STATUS.md, and the morning report lists it with the night's FEEDBACK.md lines; exp/INDEX.md archives before 100 lines, where a Contents list would become due, instead of past the 150-line limit.

## 2026-10-01: knowledge/ in lx-work
- What onboarding step 2 learned beyond the timed path (who calls the model, what the outputs mean, the exact contract, how production builds and runs it) lived only in the chat, so each compaction or new session read the code again. The skeleton now has knowledge/: INDEX.md with the writing rules (one tagged, sourced conclusion per line; open questions per topic; the commits it was read at), and business.md, contract.md, and deployment.md. Step 2 writes there while reading, step 3 records the user's answers there, and PROJECT.md and STATUS.md point there. An lx-work made earlier copies it from the skeleton.
- The checker requires knowledge/INDEX.md and keeps the topic files' headings, like the other skeleton docs.
- Changed during a project instead of at wrap-up, at the user's request, so the next project starts with it.

## 2026-09-30: drift checks from a docs review
- MAP.md drift: a pilot's time breakdown still showed the shares from before five delivered experiments, so it pointed at steps already solved. MAP.md now carries `- Measured on: baseline <commit> + exp up to NNN`; the checker holds the lx-work commit once a kept or delivered experiment past NNN changed a file MAP.md names, and checks that every `path:function` pointer still resolves. `exp.sh delivered` renames the stamp's commit, since the tree does not change.
- `hooks/test_checks.py` breaks one thing per check in clean fixtures and expects the checker to fail; a check that never fires looks like one that passes. The pre-commit hook runs it.
- Playbook counts: `✓N ✗M` in playbook/INDEX.md was maintained by hand and counted inconsistently (some cards counted pilot reruns, some did not). Evidence lines now start with ✓ or ✗ and the checker recounts them; gpu-preprocessing became ✓3 and numpy-temporaries ✓2.

## 2026-09-25: cloud repo
- The main copy has a cloud `origin`. The user decided: during a project, updates sync only locally (clone and main copy); the main copy is pushed at the end of the project, in wrap-up step 5, or when the user asks. lx-work never goes there.

## 2026-09-25: no line limit for scripts
- The user lifted the script limit: some scripts are long by nature, and splitting exp.sh would spread shared functions across files. Docs keep their limits, which keep them maintainable.

## 2026-09-25: night runs
- Night run (rules/night.md, lx-work/night/): the user asked for a mode that tries every method with evidence overnight, each in its own directory, reproducible and applicable to the project at any time, for the user to pick from in the morning. It is the one autonomous mode and lands nothing: `exp.sh try` starts a trial on the current tree, `park` saves its verified patch and reverses it, and `adopt` turns a parked trial into an experiment, applying the patch when the base has not moved. `export night/NNN` rebuilds a trial from its own base. The SessionStart hook names any experiment or trial left running.
- `exp.sh delivered`: after the user committed the kept patches, `exp.sh` refused every command, because HEAD was no longer the baseline. It now checks that HEAD is exactly the baseline plus the kept patches, makes HEAD the new baseline, and marks those experiments delivered. `export NNN` rebuilds from the item's own Base line, so a delivery does not change it.
- The skeleton PROJECT.md kept the `<rules-version>` token that setup.sh fills; the previous rename had broken it.
- From the first real night run: the queue leaves out directions DECISIONS.md rules out (a trial of a ruled-out precision change would have been queued), less code at equal speed may be parked, and run.sh writes next to itself (a copied run.sh wrote into the trial's directory after adopt). The adapter docstring no longer ties candidates to exp/.

## 2026-09-25: review after the pilots
- The bench A/A mode is built in (`--ab baseline`); a pilot had to write its own control candidate. The noise floor is per scenario (`adapter.NOISE_MS` is a dict), as the rules already required; an A/B without one exits. Under three rounds the A/B gives no verdict: an identical-code A/B with one round read as a 38 ms loss. Rounds swap which side runs first.
- The checker flags template placeholders left in STATUS.md or PROJECT.md after onboarding; a pilot finished with one in STATUS.md. "Rules version" became "Rules version at onboarding": later pulls do not update it.
- The verify and cpu-profile templates flush stderr before redirecting it, as the bench does.

## 2026-09-25: pilot, second project
- Generic bench, verify, and cpu-profile in templates/tools/, with an adapter skeleton that holds everything project-specific. The second project would otherwise rewrite the first project's tools and could repeat its measurement mistakes.
- Onboarding checks that weights are real files, not Git LFS pointers. The verify template refuses to overwrite frozen references: once optimized code lands, freezing again would hide every change.

## 2026-09-24: pilot on a real project
A pilot re-optimized an image CNN service from its unoptimized commit (6 experiments, 5 kept, -58% at L0) to find what the rules got wrong.
- Onboarding asks a few plain questions with proposed answers; the MAP.md template has a place for the stage flow; the adapter loads candidates too.
- Benchmarking: inline steps split with a function-level profiler; native logs redirected at the file-descriptor level; an A/B mode with one fresh process per scenario, since scenarios sharing a process shifted each other by several ms and produced a fake gain; noise floor per scenario from A/A runs, re-measured after large gains; targeted tests for branches the scenarios never reach; A/B medians are the recorded numbers.
- Templates: per-scenario numbers with a main scenario, no `<metric>` placeholders, lx-work ignores Python caches (8 .pyc files had been committed).
- Checker: project code in change.patch and candidate/ keeps its language (exp.sh keep produced a patch the English check rejected); finished notes cannot keep placeholders; template-shape errors also cover fields added by newer rules.
- Wrap-up commits kept patches from a temporary index; the rules repo refuses rule changes without a CHANGELOG entry, because the pilot's own rule fixes went unlogged.
