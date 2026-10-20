---
tier: once
read_when: onboarding step 2 in a discussed onboarding (STATUS.md has `- Mode: discussed`), before the baseline
---
# Onboarding step 2: the discussion

Explain your understanding first, the issues found in step 1 apart from the items to decide. Then one or two items at a time in plain words: what it means, its options, which you would pick and why; the user decides. Never the whole list at once, never a quick yes; items the code or `local/site.md` answers need only a confirmation.
Record the agreed parameters in PROJECT.md, each decision with its reason in DECISIONS.md, and what the user explains about callers, contract, or deployment in the knowledge files, where an answer replaces its open question. They belong to this project only.
- End-to-end scope: where timing starts and stops.
- Workload: input shapes, batch sizes, request pattern, warm or cold start.
- Metric: latency by default; throughput or memory only if the user says so. Every result is compared with the baseline (`rules/writing.md`, Glossary): the baseline commit's own bench unless the user names another source.
- Bench time budget for one run: it sets WARMUP and REPEAT in the adapter and goes under Metric.
- Accuracy standard (`rules/quality.md`): the identity reference, the quality baseline and its metric, the thresholds of L0 to L3 or none, and the accepted level.
- Contracts: the public interface downstream code relies on (argument order, return structure, channel order, sizes), the pre/post tensor contract (dtype, range, layout, color order, interpolation, rounding), and who consumes the output. Changing one needs the downstream impact stated first.
- Deployment: how production runs it (image and versions, how it loads the repo and resolves weights, workers per GPU, MPS), the maximum input, and what a delivery must include (CI steps that rebuild engines, dependency locks, runtime libraries). Align the local environment with production's versions where possible. A service repo pinning a separate algorithm repo ends its delivery with a merge and a new pin (`rules/wrap-up.md`).
- Done when: the target, diminishing returns (three experiments in a row inside the noise floor), or a deadline.
- Target hardware: if production is not this machine, gains here may not transfer; say so and agree how to confirm them. Read the production GPU's doc (`hardware/rtx-5090.md`, `hardware/l20.md`).
- Constraints: code that must not change, allowed dependencies, techniques ruled out (pinned memory), and the private terms from step 1, confirmed and completed.
- Environment: which Python environment the tools run in, by default the one `local/site.md` names. A package the project's code imports, or its CI or init scripts install, is added without asking, pinned to the target image's version (`local/site.md`); if pip would change an installed package's version, leave it to the user, since every project shares that environment. Any other package is a new dependency.

Once the user has confirmed, take the baseline (`rules/onboarding.md`, The baseline).
