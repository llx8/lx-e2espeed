---
tier: once
read_when: the user says the project or a milestone is done
---
# Wrapping up

Set the phase in STATUS.md to `wrap-up` first, so a later session resumes here.
Track the steps with this checklist in your reply:

    - [ ] 1 Deliver  - [ ] 2 Close  - [ ] 3 Cards  - [ ] 4 Rules review  - [ ] 5 Push

## 1. Deliver
- Run `lx-e2espeed/exp.sh check`: the project tree must equal the baseline commit plus the kept patches. Then run verify and bench, and record the numbers in STATUS.md.
- The final change against the baseline is a conclusion: bench best and the baseline in one session, alternating three runs each, the baseline in its own environment (`rules/benchmarking.md`, Inputs). Report it beside the descriptive changes recorded during the project.
- Structure pass: read the whole diff against the baseline commit once more, for code that no longer earns its place and for the shape the kept patches left behind. Refactor when you find:
  - two implementations of one step (the old path kept beside the new, a CPU copy next to the GPU one), or a switch or environment variable with one production value;
  - experiment scaffolding (timing prints, variant flags) that landed, or derivations or measurements in comments;
  - one file that now holds the orchestration, the kernels, and the reference implementation, or a function that three or more kept patches changed in turn.
  The refactor is one more experiment, before delivery: verify, a bench that shows equal speed (`rules/experiment.md`, Land and decide), its own patch and its own commit, so the record stays complete. A module someone else maintains gets a proposal to its owner, not a refactor.
- Run the handoff checks in `rules/delivery.md` that the Deployment section of PROJECT.md calls for.
- The order of what follows, and where the user's words stop (told to stop before the load test, three sessions in one night merged the service layer too, 2026-10-21):
  1. Commit: each kept patch as its own commit on its repo's working branch (below), then `exp.sh delivered`.
  2. Algorithm merge: the algorithm's `lx-dev` into its merge target, the service's dependency file pinned to that merge and committed on the service's working branch; push both working branches.
  3. Load test (next item).
  4. Service merge: the service's working branch into its production branch.
  "Commit and merge" means steps 1 and 2. "Up to the step before the load test" stops after step 2, and so does every request that does not name the service merge. Step 4 waits until the load test passed and the user, told its result, says to merge the service layer; record both in STATUS.md as `- Load test: passed <date>, <summary>` and `- Service merge: go-ahead "<the user's words>"`.
- Load-test before the service merge, on the RTX 5090 only, with the GPU free: build the delivered image and the baseline commit's image the way CI builds them (Git LFS files pulled, or the service exits at start), copy `templates/tools/load-test/` to `lx-work/tools/`, fill its TODOs, and run it with both images. Operations' load tests had caught what no single-request bench shows: no MPS, a CPU-bound worker, memory that grows with the worker count.
  It passes with no errors, hangs, or mismatches at any level; delivered throughput at every level at least the baseline's (a shortfall under 5% is rerun once); an MPS client per worker; GPU memory within the card at the peak worker count. A failure blocks the service merge. Its summary lines go in STATUS.md and the delivery facts.
  The question is how well the algorithm uses the server, not production's worker count (operations choose that from the curve): for each image the worker count climbs (1, 2, 4, 8, 12, 16; the tool mounts the image's config with each count) until the throughput stops growing, with as many clients as workers and twice that at each count; report the curve and the count where each image peaks.
  At the delivered image's peak one side must be saturated: the GPU (utilization 85% or more, or power 90% of its cap), a hardware decoder at 90% of its standalone ceiling, a worker process near 1.0 core, or all host cores. When none is, the test fails: find the wait (nsys on a worker: locks, synchronizes, the decode queue, CPU preprocessing) and record the cause before the merge.
  A hardware decoder (NVJPG, NVDEC) is a side of its own: measure its decodes/s at the service's inputs on the same GPU standalone with several processes first (nvidia-smi does not show its load reliably) and fill the tool's DECODER_CEILING_RPS.
  The client runs in several processes; a level where one of them nears a full core measured the client, not the service: it is invalid and rerun with more client processes.
  GPU memory at the peak must stay within 20% of the one-worker peak times the worker count and must not grow with the level; otherwise find the allocation that grows (pools, per-request allocations, graphs) before the merge. The tool prints these verdicts.
  One run covers everything, about 35 to 40 minutes: both images through the climb, then a soak (the delivered image at its peak count holds the top level 10 minutes: GPU memory and RAM at the end within 20% of the start), worker restarts (new PIDs) at zero, the container log free of tracebacks and of a missing MPS daemon, and GPU memory per process recorded. A rerun repeats only the part that failed.
- The kept changes sit uncommitted in the project's working tree. Show the user the diff and the final numbers, and offer a fresh-context review (for example `/code-review`).
- Before anything is called ready to push, update every doc the change made stale, in the project and in each delivered repo (a sibling's readme included): "update the docs" means all of them.
- The user commits and pushes under the company's rules. Each kept patch can become its own commit, with the change and its measured gain in the message, in the project's commit style.
- When a service repo pins a separate algorithm repo in a dependency file (repo, branch, commit), the delivery is not done until the algorithm's `lx-dev` branch is merged and the service pins the merge. Otherwise production keeps building the old algorithm code, or code from a dev branch. This is step 2, before the load test; the service's own merge is step 4. Merges, new branches, and pushes still follow the company's rules, each with the user's go-ahead.
  - Merge target: the branch `lx-dev` was created from (PROJECT.md, Working branch), whatever its name. If the algorithm repo has only its main branch (master or main, no develop), first create develop from the main branch and merge into develop; the main branch stays as it was.
  - Then set the algorithm line of the service's dependency file to that branch (develop in the second case) at its newest commit, the merge.
- When the user asks you to make those commits, apply each change.patch in number order to a temporary index (`GIT_INDEX_FILE=<tmp> git apply --cached`) and commit from it, so the working tree is never touched; afterwards `git status` is clean. Do the same with each change.<dir>.patch in its linked repo.
- Once the kept patches are committed, run `exp.sh delivered`: it checks that HEAD is exactly the baseline commit plus the kept patches, and each linked repo's HEAD its commit plus its kept patches, makes HEAD the new baseline commit and the linked HEADs the new linked commits (the baseline stays), and marks those experiments delivered, so work can go on after a milestone.

## 2. Close the record
- List known risks, ideas not tried, and parked trials the user did not adopt under "Next" in STATUS.md, so a session that reopens the project starts from them.
- Tell the user the outcome in your reply, with the facts `rules/delivery.md` lists for a delivery, each kept change with its gain and difference level, and the risks. STATUS.md, exp/INDEX.md, and DECISIONS.md are the full record; write no separate report into lx-work.
- State gains honestly: gains from different rows or dates do not add up, a comparison across dates is descriptive rather than a paired A/B, and a hardware or driver change is not a software gain.

## 3. Distill cards
Cards go straight into lx-e2espeed, which every project shares, so the playbook you search is always the newest; edit and commit it as `rules/writing.md` says.
For each kept experiment, and each dropped one that taught something:
- Search playbook/INDEX.md for the same technique. If a card exists, update its effect range, conditions, and evidence instead of adding a card.
- Each Evidence line is one data point: ✓ when the change reached production (delivered at this wrap-up), ✗ when it was tried and did not (no gain, failed verify, dropped, or parked and passed on by the user), with the reason. A rerun of an earlier project counts as its own line. Update the index line's `✓N ✗M` to match: the next standard pass ranks by it.
- A delivered change with no card gets one; so does a failure that teaches where a technique does not fit.
- Otherwise create a card from `templates/card.md` in the category where the time was spent, and add its line to the index.
- Scrub everything project-specific: project, company, model, and file names, paths, project code, business details. Describe the model generically, for example "decoder-only transformer, batch 1".
- A failure only shows the technique did not fit this project. Record it in the card's "Does not apply" as the condition that made it fail (bottleneck, model type, shapes, hardware), and in Evidence as one more data point. Never delete a technique for failing once.

Facts other projects of the company share (images, CI, the service framework, tested libraries, reference commits) get a line in `local/site.md` pointing to this project's record; it stays on this machine.

## 4. Review the rules
Start from lx-work/FEEDBACK.md. Then ask: where did the work drift from these rules, what was missing, which docs were never read?
Propose changes to the user. Apply the approved ones, log them in CHANGELOG.md, and delete the FEEDBACK.md entries they resolve.

## 5. Push to the cloud
lx-e2espeed is one repo on this machine, and its `origin` is the cloud repo. Rule changes were committed as they were made (`rules/writing.md`), so every project already has them. Pushing needs no go-ahead (the user, 2026-10-20): push after any finished change to lx-e2espeed, and at the latest in this step.
- Run the checker in lx-e2espeed. If it fails on files you did not change, another session is midway through a change: ask the user.
- Push: `git -C lx-e2espeed push origin main`. If the cloud has commits lx-e2espeed lacks, pull them first (`git -C lx-e2espeed pull --ff-only origin main`), run the checker again, and ask the user if it cannot fast-forward.
- Set the phase in STATUS.md to `done` at the end of the project, or back to `optimizing` after a milestone, and commit lx-work. It never goes to the lx-e2espeed cloud repo, and stays local unless the user sets up a remote of its own.
