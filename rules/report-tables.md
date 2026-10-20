---
tier: on-demand
read_when: before a project report, a stage-time table, or a summary or weekly report for others
---
# Report tables and written reports

The forms the user settled for reports; the reply rules in `rules/reporting.md` hold too.

## Stage times
Stage times, production against the optimized version with each operation's effect on the outputs, go in one table (the user's final form, after several layouts failed):
- Six columns: layer with its time "production → optimized"; operation, a few words; production time; what was optimized, short phrases joined by semicolons, or "none"; optimized time; result change.
- Result change: the distance to the quality baseline (`rules/quality.md`) at P50 over the input set, "before → after" in the project's quality metric and unit (PROJECT.md), on a row whose change moves outputs; otherwise "0 (unchanged code)" or "0 (bitwise identical)". Numbers only, never "slight", "better", or "faster".
- Rows: the five layers in path order, service pre-processing, algorithm pre-processing, inference, algorithm post-processing, service post-processing; a layer's name and time only on its first row.
- Then a bold total row with the end-to-end median only, and bold quality rows P50, P90, P99 of the distance to the quality baseline, whose last cell is the baseline's own seed spread at that percentile; the "what was optimized" cell stays empty there.
- Keep it narrow: the terminal turns a table that does not fit into key-value blocks. Two or three lines under it say the GPU, the inputs, what the quality metric measures, the reference seed, and which rows were measured now or come from earlier bitwise checks.
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

