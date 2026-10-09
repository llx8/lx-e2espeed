---
tier: once
read_when: the user says the project or a milestone is done
---
# Wrapping up

Set the phase in STATUS.md to `wrap-up` first, so a later session resumes here.
Track the steps with this checklist in your reply:

    - [ ] 1 Deliver  - [ ] 2 Close  - [ ] 3 Cards  - [ ] 4 Rules review  - [ ] 5 Sync

## 1. Deliver
- Run `lx-e2espeed/exp.sh check`: the project tree must equal the baseline commit plus the kept patches. Then run verify and bench, and record the numbers in STATUS.md.
- The final change against the baseline is a conclusion: bench best and the baseline in one session, alternating three runs each, the baseline in its own environment (`rules/benchmarking.md`, Inputs). Report it beside the descriptive changes recorded during the project.
- Read the whole diff once more for code that no longer earns its place. A simplification is one more experiment, so the record stays complete.
- Run the handoff checks of the `gpu-inference-delivery` skill that the Deployment section of PROJECT.md calls for.
- The kept changes sit uncommitted in the project's working tree. Show the user the diff and the final numbers, and offer a fresh-context review (for example `/code-review`).
- The user commits and pushes under the company's rules. Each kept patch can become its own commit, with the change and its measured gain in the message, in the project's commit style.
- When a service repo pins a separate algorithm repo in a dependency file (repo, branch, commit), the delivery is not done until the algorithm's dev branch is merged and the service pins the merge. Otherwise production keeps building the old algorithm code, or code from a dev branch. The user's rule (2026-10-09); merges, new branches, and pushes still follow the company's rules, each with the user's go-ahead.
  - Merge target: the branch the dev branch was created from, whatever its name. If the algorithm repo has only its main branch (master or main, no develop), first create develop from the main branch and merge into develop; the main branch stays as it was.
  - Then set the algorithm line of the service's dependency file to that branch (develop in the second case) at its newest commit, the merge.
- When the user asks you to make those commits, apply each change.patch in number order to a temporary index (`GIT_INDEX_FILE=<tmp> git apply --cached`) and commit from it, so the working tree is never touched; afterwards `git status` is clean.
- Once the kept patches are committed, run `exp.sh delivered`: it checks that HEAD is exactly the baseline commit plus the kept patches, makes HEAD the new baseline commit (the baseline stays), and marks those experiments delivered, so work can go on after a milestone.

## 2. Close the record
- List known risks, ideas not tried, and parked night trials the user did not adopt under "Next" in STATUS.md, so a session that reopens the project starts from them.
- Tell the user the outcome in your reply, with the facts the `gpu-inference-delivery` skill lists for a delivery, each kept change with its gain and difference level, and the risks. STATUS.md, exp/INDEX.md, and DECISIONS.md are the full record; write no separate report into lx-work.
- State gains honestly: gains from different rows or dates do not add up, a comparison across dates is descriptive rather than a paired A/B, and a hardware or driver change is not a software gain.

## 3. Distill cards
First bring the clone up to date with `git -C lx-e2espeed pull --ff-only`, so new cards merge into the newest playbook instead of duplicating it.
For each kept experiment, and each dropped one that taught something:
- Search playbook/INDEX.md for the same technique. If a card exists, update its effect range, conditions, and evidence instead of adding a card.
- Each Evidence line is one data point, marked ✓ (worked) or ✗ (failed); a rerun of an earlier project counts as its own line. Update the index line's `✓N ✗M` to match; the checker recounts it.
- Otherwise create a card from `templates/card.md` in the category where the time was spent, and add its line to the index.
- Scrub everything project-specific: project, company, model, and file names, paths, project code, business details. Describe the model generically, for example "decoder-only transformer, batch 1".
- A failure only shows the technique did not fit this project. Record it in the card's "Does not apply" as the condition that made it fail (bottleneck, model type, shapes, hardware), and in Evidence as one more data point. Never delete a technique for failing once.

## 4. Review the rules
Start from lx-work/FEEDBACK.md. Then ask: where did the work drift from these rules, what was missing, which docs were never read?
Propose changes to the user. Apply the approved ones, log them in CHANGELOG.md, and delete the FEEDBACK.md entries they resolve.

## 5. Sync to the main copy, then the cloud
The main copy is the clone's `origin` (`git -C lx-e2espeed remote get-url origin`), a directory on this machine; its own `origin` is the cloud repo. During a project, rule updates move only between the clone and the main copy. The cloud gets them at the end of the project, in this step, or when the user asks; a milestone wrap-up skips the push.
- Show the user the full diff of lx-e2espeed. Commit in the clone only after approval.
- Bring the change into the main copy by pulling from its side: `git -C <main copy> pull --ff-only <project>/lx-e2espeed main`. Pull, not push: git refuses a push into a repo whose branch is checked out.
- If the main copy has uncommitted changes, stop and ask the user.
- If the pull cannot fast-forward, another project synced first. Run `git -C lx-e2espeed pull --rebase`, merge any card both projects touched, run the checker, show the user the result, then pull into the main copy again.
- Other project clones pick the change up with `git -C lx-e2espeed pull --ff-only`, then sync their lx-work (`rules/naming.md`).
- At the end of the project, run the checker in the main copy, then push it: `git -C <main copy> push origin main`. If the cloud has commits the main copy lacks, pull them into the main copy first (`git -C <main copy> pull --ff-only origin main`), run the checker again, and ask the user if it cannot fast-forward.
- Set the phase in STATUS.md to `done` at the end of the project, or back to `optimizing` after a milestone, and commit lx-work. It never goes to the lx-e2espeed cloud repo, and stays local unless the user sets up a remote of its own.
