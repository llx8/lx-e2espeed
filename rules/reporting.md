---
tier: on-demand
read_when: before a reply that reports results, puts numbers in a table, answers a question about the work, or explains how something runs; before a project report or a summary for others
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

## Stage times
Stage times, production against the optimized version with each operation's effect on the outputs, go in one table, the user's final form (2026-10-18), with exactly the columns and rows below.
- Never one of the layouts it replaced, each rejected by the user: rows sorted by time instead of the path; an "inference" row that is the unmeasured remainder; indented rows; one table per layer; seven or more columns (the terminal broke them into a list); words like "slightly" or "better" in cells; numbers from another version, machine, or image.
- Six columns: layer with its time "production → optimized"; operation, a few words; production time; what was optimized, short phrases joined by semicolons, or "none"; optimized time; result change.
- Result change: the distance to the quality baseline (`rules/quality.md`) at P50 over the input set, "before → after" in the project's quality metric and unit (PROJECT.md), on a row whose change moves outputs; otherwise "0 (unchanged code)" or "0 (bitwise identical)". Numbers only, never "slight", "better", or "faster".
- Rows: the five layers in path order, service pre-processing, algorithm pre-processing, inference, algorithm post-processing, service post-processing; a layer's name and time only on its first row.
- Then a bold total row with the end-to-end median only, and bold quality rows P50, P90, P99 of the distance to the quality baseline, whose last cell is the baseline's own seed spread at that percentile; the "what was optimized" cell stays empty there.
- Keep it narrow: the terminal turns a table that does not fit into key-value blocks. Short cells, a few words each. Two or three lines under it say the GPU, the inputs, what the quality metric measures, the reference seed, and which rows were measured now or come from earlier bitwise checks.
- Measure both versions now, on one GPU, each in its own deployment image and settings (driver capabilities, decoder, engine built there), stage mode timing every operation. Never numbers from an older version, another machine, or another environment, even marked.
- A row's result change comes from an intermediate state that changes only that row (production with only the decoder change, the optimized tree with one change reverted), against a reference seed other than any fixed random draw in the optimized model; a row bundling changes gives their sum.
- The input set: at least 100 production-like inputs; with 100, P99 is about one input's value, so say so before drawing on it.
- Inputs whose paths differ only in decoding share the table; the other decoder takes one row under it.

## Project report
At the end of a pass or a milestone, the first answer to "how does it run", and whenever the user asks for a report on a project. The user learns the project from it, so it opens with the brief, answered from MAP.md in this order (the user's order), a few short lines each, on the delivered version:
1. What the service receives and what it returns.
2. Service pre-processing, then algorithm pre-processing: the operations in order, with device, dtype, and shape.
3. The model: its inputs and outputs (name, shape, dtype, meaning), its network structure, and its notable operations.
4. The top kernels from Nsight Systems: the five largest by share of GPU time, each with its share and the operation it belongs to.
5. Algorithm post-processing, then service post-processing, as in 2.

Then:
- The stage table for the main scenario, both versions measured now; the decode row names its decoder; its lines underneath say what the quality metric measures and against what.
- Then what is wrong or risky, and the user's open decisions, short.

## Summaries for others
- Credit colleagues inside the text where their work is used (a plugin library and the gain it brought), never in a separate block.
- Speed and output quality in separate tables, holding only the options being decided; every cell a number or where it was measured, no empty or dash cells.
- Show a quality gap as counts per error band, with a hard input split out, not as one median.
- A weekly report: a few short prose paragraphs in the user's language, no bullets, lists, tables, or sections; in order the task, the project (input, processing, output), the model, the optimizations, the results with only the headline numbers, the open problem; one or two sentences each for the project and the model. No space between Chinese text and numbers, units, or English terms.

## While the user watches
During a long run the user watches, send a short status line at each step and before each long wait; silence reads as a stall. A run the user is away from reports at its end instead (`rules/self-iteration.md`).
