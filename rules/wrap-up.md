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
- Read the whole diff once more for code that no longer earns its place. A simplification is one more experiment, so the record stays complete.
- Run the handoff checks in `rules/delivery.md` that the Deployment section of PROJECT.md calls for.
- The kept changes sit uncommitted in the project's working tree. Show the user the diff and the final numbers, and offer a fresh-context review (for example `/code-review`).
- Before anything is called ready to push, update every doc the change made stale, in the project and in each delivered repo (a sibling's readme included): "update the docs" means all of them.
- The user commits and pushes under the company's rules. Each kept patch can become its own commit, with the change and its measured gain in the message, in the project's commit style.
- When a service repo pins a separate algorithm repo in a dependency file (repo, branch, commit), the delivery is not done until the algorithm's dev branch is merged and the service pins the merge. Otherwise production keeps building the old algorithm code, or code from a dev branch. The user's rule (2026-10-09); merges, new branches, and pushes still follow the company's rules, each with the user's go-ahead.
  - Merge target: the branch the dev branch was created from, whatever its name. If the algorithm repo has only its main branch (master or main, no develop), first create develop from the main branch and merge into develop; the main branch stays as it was.
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
- Each Evidence line is one data point, marked ✓ (worked) or ✗ (failed); a rerun of an earlier project counts as its own line. Update the index line's `✓N ✗M` to match; the checker recounts it.
- Otherwise create a card from `templates/card.md` in the category where the time was spent, and add its line to the index.
- Scrub everything project-specific: project, company, model, and file names, paths, project code, business details. Describe the model generically, for example "decoder-only transformer, batch 1".
- A failure only shows the technique did not fit this project. Record it in the card's "Does not apply" as the condition that made it fail (bottleneck, model type, shapes, hardware), and in Evidence as one more data point. Never delete a technique for failing once.

Facts other projects of the company share (images, CI, the service framework, tested libraries, reference commits) get a line in `local/site.md` pointing to this project's record; it stays on this machine.

## 4. Review the rules
Start from lx-work/FEEDBACK.md. Then ask: where did the work drift from these rules, what was missing, which docs were never read?
Propose changes to the user. Apply the approved ones, log them in CHANGELOG.md, and delete the FEEDBACK.md entries they resolve.

## 5. Push to the cloud
lx-e2espeed is one repo on this machine, and its `origin` is the cloud repo. Rule changes were committed as they were made (`rules/writing.md`), so every project already has them. The cloud gets them at the end of the project, in this step, or when the user asks; a milestone wrap-up skips the push.
- Run the checker in lx-e2espeed. If it fails on files you did not change, another session is midway through a change: ask the user.
- Push: `git -C lx-e2espeed push origin main`. If the cloud has commits lx-e2espeed lacks, pull them first (`git -C lx-e2espeed pull --ff-only origin main`), run the checker again, and ask the user if it cannot fast-forward.
- Set the phase in STATUS.md to `done` at the end of the project, or back to `optimizing` after a milestone, and commit lx-work. It never goes to the lx-e2espeed cloud repo, and stays local unless the user sets up a remote of its own.
