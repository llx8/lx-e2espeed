---
tier: once
read_when: the user starts a standard pass on a project, during one, and the first session after it
---
# Standard pass

The user hands over a new project to optimize, in any words, or says to run the standard pass, and leaves or watches. Begin at once: no confirmation question, no plan to approve (the user corrected that twice).
You onboard it, measure it, work through a queue built from what past projects delivered (Step 2) without the user's picks, and report once at the end, so the routine part of every project runs without a discussion and the changes that alter outputs reach the user finished, with numbers.
It is safe for the same reason as a self-iteration run: a change lands only when its outputs are identical and its gain is measured; everything else is parked. Starting the pass approves that queue under these landing rules, and nothing beyond it.

## Contract with the user
- The pass does not stop: it runs as `rules/self-iteration.md`, During the run, says (no turn ends for a question, a progress report, or a plan; questions under "Waiting for the user" or a trial's "Needs the user" column with the default taken; the Stop hook holds the turn while the Run line names this session).
  - The Run line, under "In progress" in STATUS.md: `- Run: standard pass, session <id>, item <n> of <N>`, with the id the SessionStart hook printed; a resuming session puts its own id there first.
  - It ends when every item is kept, parked, or skipped with a measured reason; at the user's end time; or when the environment breaks (the same infrastructure error twice). A permission prompt nobody answers is the one other stop: check before the first command that the session runs without them.
- Starting the pass also covers what onboarding would ask about: `setup.sh project`, `git init` of a project that is no repo (nothing pushed), excluding weights, and linking a sibling repo the project reads code from.
- Decisions that are the user's are listed once, in the first status line, and defaulted at once, each into PROJECT.md and DECISIONS.md as `<value> (pass default, not discussed)`; an answer arriving mid-pass applies from the next item.
  - Production GPU: the deployment files, else the shared L20 (`hardware/l20.md`). Production image: the deployment files, else the previous image in `local/site.md`.
  - Calibration data: none, so INT8 and FP8 are skipped. Scenarios and shapes: the repo's own test inputs or example requests, and the contract's maximum.
  - Extra inputs: when production-like data sits on this machine, candidates in `lx-work/local/inputs/` for the user to pick (Waiting), and a fixed random sample of about 100 for every change that moves outputs.
  - Quality metric: in the output's own units as the caller uses it (`rules/quality.md`), no thresholds until the user sets them.
  - Workers: one. End time: none.
- Lands without asking (`exp.sh keep`): an item whose outputs are L0 against the frozen references, whose gain is beyond the noise floor (or less code at equal speed), and which changes no contract. L1 lands too when the project already runs that model as a built engine at that precision and the engine-noise tool measured the spread.
- Everything else with a gain is parked (`exp.sh park`) with its level and what the user must decide; numeric precision and `torch.compile` may be tried this way because they do not land.
- Stays the user's, by parking or skipping, never by stopping: a contract change, a dependency beyond the alignment `rules/onboarding.md` allows, a change to the baseline or the references, any difference beyond L0.
- Every other rule holds (`rules/experiment.md` included); rules, cards, and tolerances stay untouched, and a problem with them goes to FEEDBACK.md.

## Step 1: onboarding without the discussion
`rules/onboarding.md` step 1 as written. Step 2's items come from the code, the deployment files, and the defaults above, recorded as such. Both references of `rules/quality.md` are frozen in the pass's environment:
- Identity: the unmodified code's own outputs. A change lands only at L0 against them; verify grades every change, so the user can judge the parked ones.
- Quality baseline: the network in FP32 inside the unmodified request path, with its seed spread; production, best, and every parked item are put against it.
- Baseline speed: the unmodified code in the image production runs, on the production GPU, three runs (the run-image tool), with the time breakdown and noise floor measured there. When that GPU is not this machine, run every A/B where the harness runs and confirm there at the end.
- Metric: end-to-end latency and the stage breakdown, unless the deployment files show a throughput service.
- A shared production GPU runs other people's jobs: before each measurement there, check `nvidia-smi`, wait and retry instead of measuring beside them, record the busy periods, and lock the clock as far as permissions allow.
Then the phase is optimizing, and Step 2 builds the queue, with evidence from MAP.md as each stage's time in milliseconds, not its share (absolute times keep the decision stable when the ladder later shrinks the model's time). A stage below the noise floor is skipped with that number as the reason.

## Step 2: the queue, from what past projects delivered
The queue is not a fixed list: it comes from the playbook, which every wrap-up updates with what reached production and what did not (`rules/wrap-up.md`, step 3), so each pass starts from more experience than the last.
- For each stage above the noise floor, largest time first, take the cards in `playbook/INDEX.md` whose symptom the profile shows. Open each; skip one whose "Does not apply" condition holds here, with that condition as the reason.
- Order them by expected saving: the stage's milliseconds times the card's effect range. Between close ones the better record `✓N ✗M` (delivered against not) goes first; a card with more ✗ than ✓ is read for its ✗ lines first, which say where it fails.
- A lead the profile shows that no card covers joins by its expected saving; its outcome becomes a card at wrap-up.
- The queue goes in trials/INDEX.md, each row with its stage time, card, and record.

Fixed, whatever the ranking:
- Environment first: the project in the target image's environment (`local/site.md`), with the code changes it needs to run there and nothing more. Kept whatever the speed, since the user set the target; its speed against the production image opens the report, with its outputs' level against outputs dumped there when the run-image tool can dump them. A project already on the target image has no such item.
- Redundant work before engine work, one experiment per finding (computation repeated or unused, repeated initialization, round trips, syncs, temporaries; skill `perf-torch-sync-free`), so the ladder measures the model and not the glue.
- Engine work runs as a ladder, built and timed on the production GPU (tactics and precision support differ per compute capability); a project already on a TensorRT engine skips the rungs it has.
  - Rungs: FP32 (it separates the graph optimization's gain from the precision's), FP16, then INT8 and FP8 only with calibration data, run on a bounded sample under the memory cap with its peak estimated first. FP8 only through an explicit quantization flow, worth it for transformer blocks; a CNN takes INT8.
  - Each rung uses the production builder settings (`playbook/env/trt-builder-level.md`) and shape policy (`playbook/serving/dynamic-batch-profiles.md`), and passes the MPS scan (`playbook/serving/mps-grid-barrier.md`) before it is parked or kept.
  - On the production precision (FP16 when production has no engine yet), the ranked compute and host cards (layer cleanup, plugins, a CUDA graph of the enqueue) run on top of it.
  - When that precision's distance to the quality baseline exceeds the seed spread, find the smallest group of layers that closes it (card `engine-difference-triage`, Pinning) and park pinning it at build.
- A decoder's outputs usually differ from the CPU library's, so GPU or hardware decode is normally parked with its level. A stage left on the CPU gets a line in NOTE.md with its time and the transfer it would cost.
- `torch.compile` is the last and least promising trial (`playbook/compute/torch-compile.md`).
Kept items stack: each starts from best, so the chain is the delivered code. Parked items start from the current best too and are measured against it. The engine ladder is the one place where parked trials build on each other: each rung's patch holds the whole engine change up to it.
Every item runs as `rules/experiment.md` steps 3 to 5 (`exp.sh new`) or as a trial of `rules/self-iteration.md` (`exp.sh try`), with Evidence, Hypothesis, and Plan written as if the user would read them: every choice needs enough reason behind it. Timebox: 90 minutes of your own work per item, a build or calibration what its estimate says; past it, park what works or drop it with what was learned, and queue the rest as a day task.
Last: the structure pass of `rules/wrap-up.md` over the kept chain, landed as one more experiment when it finds something; and when the production GPU is not this machine, the kept chain and every parked rung benched there, three runs each, in the image production runs. An item that loses its gain there is reported as such, and the user decides.

## Ending and the report
End as a self-iteration run ends: `exp.sh check` passes, "In progress" is none (the Run line replaced), "Waiting for the user" says `standard pass <date>: N kept, M parked, K skipped; report it (rules/standard-pass.md)`, lx-work committed.
The report is one reply, laid out as `rules/self-iteration.md` describes, opened by the project brief and tables of `rules/reporting.md` (Project report), plus: the kept chain as one table in order with each item's gain, level, and quality against the quality baseline; the ladder as rungs with cumulative numbers; one line per skipped item with its measurement; the defaults taken; the environment item's numbers; the production-GPU confirmation.
The user then adopts parked items (`exp.sh adopt`; telling you to use one is its adoption), changes defaults, or says the project is done; from there the ordinary loop applies.
