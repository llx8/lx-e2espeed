---
tier: on-demand
read_when: before a reply that reports results, answers a question about the work, or explains how something runs; before a summary for others
---
# Reporting to the user

The user decides from your replies, not from the files; most corrections in the first projects were about replies, and several made the user angry.

## Answering
- Open a report on work with its state in one sentence: what is done, where it is stuck, the one thing needed from the user, if any; details after.
- Before asking the user anything, look it up: `local/site.md`, the deployment files, PROJECT.md and DECISIONS.md, the defaults of `rules/standard-pass.md`. Ask only what none of them answers.
- Answer what was asked, and only that; anything else takes one line at the end, or stays out. Report what is wrong or risky and leave out what is fine.
- Call each project by the user's own name for it and say which service an output belongs to. Name an image by its name in the folder the user works from.
- When a question or go-ahead can mean two things (which sides to compare, speed or quality), or read literally is already done or rests on a premise the records contradict, ask which in one line before measuring.
- Follow settled decisions (DECISIONS.md, PROJECT.md, this session); never reopen them, and check an item there before listing it as open. A decision stands when a later measurement moves a number it weighed: carry it out and state the new number in one line.
- Work from the inputs the user agreed stand for production (PROJECT.md, Workload).
- How code or configuration behaves is answered from the code and configuration; reach live systems (CI runners, servers, remote APIs) only when the user asks.
- Explain an option by what it means for the user, the mechanics only when asked. Name runs, builds, and settings in the user's terms ("fp32, TF32 on", not a flag), say in words what a build sets before running it, and explain a technical term where it first appears.
- FP32 means the framework's default (TensorRT without the FP16 flag, TF32 on; PyTorch's defaults): test that one variant unless the user names another.
- When saying where steps run (CPU or GPU), say which data each step reads and writes.
- About the next phase, say what you will propose and that the user picks.
- Every conclusion comes from an experiment run for it, with enough data to carry it (quality affected or not, faster or not, a layer sensitive); never from reasoning, an older number read another way, or an expectation. Until measured, it is "not measured": a revert once called harmless to quality measured worse on the production GPU.
- When the user orders an experiment, start it; never predict its outcome instead.
- When a report lacks a number, measure it with the existing tools on the current tree first; never offer an estimate or ask whether to measure. A request for the report covers that measurement; changing code, or a run beyond the bench time budget in PROJECT.md, still needs the go-ahead.

## Results
- Lead with the verdict in one plain sentence, then the numbers.
- A difference inside the noise floor is "unchanged", never faster or slower, in words and tables alike.
- Keep the agreed baseline fixed: give a step's numbers on the baseline's machine, or say they are another machine's and why they differ.
- Report the scenario the user named as the one that matters; others on request. An agreed layout or scenario choice is a `- Reports:` line under Metric in PROJECT.md, read before any report, and kept until the user changes it (rows, columns, groups, the side compared with). Unless agreed otherwise, rows are cumulative steps from the baseline, each with its median saving, never compared with the current tree.
- When latency depends on the input size per request, split the scenario into size groups if the user asks, each with median, P90, P99, then the whole scenario.
- One point per table: no cells packed with slashes; an input that dominates a column is shown apart.

## Reports and tables
A project report, a stage-time table, and summaries or weekly reports for others follow `rules/report-tables.md`.

## While the user watches
During a long run the user watches, send a short status line at each step and before each long wait; silence reads as a stall. A run the user is away from reports at its end instead (`rules/self-iteration.md`).
