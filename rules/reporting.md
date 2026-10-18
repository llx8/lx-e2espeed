---
tier: on-demand
read_when: before a reply that reports results, answers a question about the work, or explains how something runs; before a summary for others
---
# Reporting to the user

The user decides from your replies, not from the files. Most corrections in the first projects were about replies, not about the work itself (their FEEDBACK.md, 2026-10-06 to 2026-10-12), and several made the user angry.

## Answering
- Open a report on work with its state in one sentence: what is done, where it is stuck, and the one thing needed from the user, if any; details after (the user, 2026-10-16: a long onboarding report left them asking whether it was done).
- Before asking the user anything, look it up: `local/site.md`, the deployment files, PROJECT.md and DECISIONS.md, and the defaults of `rules/standard-pass.md`. Ask only what none of them answers.
- Answer what was asked, and only that. Status, side findings, and proposals the question did not ask for stay out, or take one line at the end. A long report of everything done reads as work nobody asked for.
- Report what is wrong or risky and leave out what is fine (the user, 2026-10-13, in four projects).
- Call each project by the user's own name for it and say which service an output belongs to. Name an image by its name in the folder the user works from, not by a source index.
- When a question or a go-ahead can mean two things (which two sides to compare, speed or output quality), ask which in one line before measuring. A wrong guess costs a whole run. Ask the same way when a request, read literally, is already done or rests on a premise the records contradict.
- Follow settled decisions (DECISIONS.md, PROJECT.md, what the user said in this session); never ask them again as open questions. Extra detail about one is information, not a question. Check an item there before listing it as open.
- A decision stands when a later measurement moves a number it weighed: carry it out and state the new number in one line, without asking again.
- Work from the inputs the user agreed stand for production (PROJECT.md, Workload); do not open questions they already answer.
- A question about how code or configuration behaves is answered from the code and configuration; reach live systems (CI runners, servers, remote APIs) only when the user asks (the user, 2026-10-12: "reason from the code alone").
- Explain an option by what it means for the user; the mechanics only when asked.
- Name runs, builds, and settings in the user's terms ("fp32, TF32 on", not an implementation flag), and say in words what a build sets before running it. Explain a technical term in plain words where it first appears.
- FP32 means the framework's default (TensorRT without the FP16 flag, TF32 on; PyTorch's defaults): test that one variant unless the user names another.
- When saying where steps run (CPU or GPU), say which data each step reads and writes: a CPU step on metadata otherwise reads as the data going back to the CPU.
- About the next phase, say what you will propose and that the user picks.
- When a report lacks a number, measure it with the existing tools on the current tree before reporting, never offer an estimate or ask whether to measure (the user, 2026-10-11: "measure what is missing first"). A request for the report covers that measurement; changing code, or a run beyond the bench time budget in PROJECT.md, still needs the go-ahead.

## Results
- Lead with the verdict in one plain sentence, then the numbers.
- A difference inside the noise floor is "unchanged", never faster or slower, in words and tables alike (the user, 2026-10-17, on a "0.02 ms faster" engine).
- Keep the agreed baseline fixed: give a step's numbers on the baseline's machine, or say they are another machine's and why they differ.
- Report the scenario the user named as the one that matters; other scenarios on request. An agreed report layout or scenario choice is a `- Reports:` line under Metric in PROJECT.md, read before any report.
- Keep an agreed report layout until the user changes it: its rows, columns, groups, and the side it compares with. Unless agreed otherwise, rows are cumulative steps from the baseline, each with its median saving; never compare with the current tree instead (the user, 2026-10-12).
- When latency depends on the input size per request (objects per image, sequence length), split the scenario into size groups if the user asks, each with median, P90, and P99, then give the whole scenario.
- One point per table: no cells packed with slashes, and an input that dominates a column is shown apart.

## Stage times
Every stage breakdown, in a report or on request, takes the user's layout (2026-10-18, after a table sorted by time whose "inference" row was an unmeasured remainder):
- Five layers in path order: service pre-processing, algorithm pre-processing, inference, algorithm post-processing, service post-processing. Each layer's total, then one row per operation under it saying what it does.
- Every row is measured (`rules/benchmarking.md`, Time breakdown), none is a remainder; give the stage-mode total and the normal end-to-end median beside it.
- Measure the current tree; never show an older version's stages, even marked stale.
- Inputs whose paths differ only in decoding share one table; give the other decoder's row apart.

## Project report
The end of a pass or a milestone, and the first answer to "how does it run". The user asked these one by one after a pass report that listed only items (2026-10-17); give them unasked, each short.
- Input and output: what the service receives (formats, sizes) and returns, and what the caller does with it.
- The path, one row per stage on each side (service, algorithm): what it does, where it runs and on which data, in production and in best. Decoding names its decoder (hardware, GPU, CPU library); the model names itself in a sentence (architecture, task), its engine, and precision.
- Speed: production's end to end and per stage, each input format apart (JPEG, PNG), then the same for best.
- Changes: one row per operation in order, what it was and what it became, before and after in ms, the saving, and its quality against the quality baseline.
- Quality: what the metric measures in plain words, what it is measured against (`rules/quality.md`), and production's own distance, with the seed spread beside it.
- Then what is wrong or risky, and the user's open decisions.

## Summaries for others
- Credit colleagues inside the text where their work is used (a plugin library and the gain it brought), never in a separate block.
- Speed and output quality in separate tables, holding only the options being decided.
- Every cell holds a number or says where it was measured; no empty or dash cells.
- Show a quality gap as counts per error band, with a hard input split out, not as one median.
- A weekly report: a few short prose paragraphs in the user's language, no bullets, numbered lists, tables, or sections; in order the task, the project (input, processing, output), the model, the optimizations, the results with only the headline numbers, the open problem; one or two sentences each for the project and the model.
  No space between Chinese text and numbers, units, or English terms (the user, 2026-10-10: one paragraph "only split into paragraphs, never into points", then "too long, keep it a summary").

## While the user watches
During a long run the user is watching, send a short status line at each step and before each long wait; silence reads as a stall. A self-iteration run the user is away from reports at its end instead (`rules/self-iteration.md`).
