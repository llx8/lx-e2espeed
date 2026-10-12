---
tier: on-demand
read_when: before a reply that reports results, answers a question about the work, or explains how something runs; before a summary for others
---
# Reporting to the user

The user decides from your replies, not from the files. Most corrections in the first projects were about replies, not about the work itself (their FEEDBACK.md, 2026-10-06 to 2026-10-12), and several made the user angry.

## Answering
- Answer what was asked, and only that. Status, side findings, and proposals the question did not ask for stay out, or take one line at the end. A long report of everything done reads as work nobody asked for.
- When a question or a go-ahead can mean two things (which two sides to compare, speed or output quality), ask which in one line before measuring. A wrong guess costs a whole run.
- Follow settled decisions (DECISIONS.md, PROJECT.md, what the user said in this session); never ask them again as open questions. Extra detail about one is information, not a question.
- Work from the inputs the user agreed stand for production (PROJECT.md, Workload); do not open questions they already answer.
- Explain an option by what it means for the user; the mechanics only when asked.
- Name runs, builds, and settings in the user's terms ("fp32, TF32 on", not an implementation flag), and say in words what a build sets before running it.
- When saying where steps run (CPU or GPU), say which data each step reads and writes: a CPU step on metadata otherwise reads as the data going back to the CPU.
- About the next phase, say what you will propose and that the user picks.
- When a report lacks a number, measure it with the existing tools on the current tree before reporting, never offer an estimate or ask whether to measure (the user, 2026-10-11: "measure what is missing first"). A request for the report covers that measurement; changing code, or a run beyond the bench time budget in PROJECT.md, still needs the go-ahead.

## Results
- Lead with the verdict in one plain sentence, then the numbers.
- Report the scenario the user named as the one that matters, recorded in DECISIONS.md; other scenarios on request.
- Keep an agreed report layout until the user changes it: its rows, columns, groups, and the side it compares with. Unless agreed otherwise, rows are cumulative steps from the baseline, each with its median saving; never compare with the current tree instead (the user, 2026-10-12).
- When latency depends on the input size per request (objects per image, sequence length), split the scenario into size groups if the user asks, each with median, P90, and P99, then give the whole scenario.
- One point per table: no cells packed with slashes, and an input that dominates a column is shown apart.

## Summaries for others
- Speed and output quality in separate tables, holding only the options being decided.
- Every cell holds a number or says where it was measured; no empty or dash cells.
- Show a quality gap as counts per error band, with a hard input split out, not as one median.

## While the user watches
During a long run the user is watching, send a short status line at each step and before each long wait; silence reads as a stall. A night run, where the user is away, reports in the morning instead (`rules/night.md`).
