---
tier: on-demand
read_when: locating code on the end-to-end path, choosing where to look next, or explaining how the project runs
---
# Code map

Pointers only (`path:function`). Never paste code here; pasted code goes stale. The project brief (`rules/reporting.md`, Project report) is answered from this file.

## Service contract
<What the service receives (request format, field, encoding, typical and maximum size) and returns (format, fields), in two lines; exact formats in knowledge/contract.md.>

## End-to-end path and time breakdown
Shares come from the stage timing on the tree named below; they decide where to look first.
The checker flags this map once a kept experiment changes a file it points to: re-run the stage timing on best, then rewrite the breakdown, the hot spots, the top kernels, and this line.
- Measured on: <"baseline commit" and its hash, plus " + exp up to NNN" once the timed tree includes experiments; NNN is the highest kept or delivered one in it>

<Measurement conditions, in this paragraph because the line above holds only the stamp: date, tool, environment, scenario, number of calls, SM clock, and where the raw output is.>

    <ASCII flow: each stage with its device ([CPU] or [GPU]), dtype, and shape, and every copy between them>

| Layer | Operations, in order (device, dtype, shape in → out) | Code | Share of e2e time |
|---|---|---|---|
| Service pre-processing | <decode, validation, conversion> | <path:function> | |
| Algorithm pre-processing | <resize, normalize, layout, copies to the GPU> | | |
| Inference | <the model call: framework or engine, precision> | | |
| Algorithm post-processing | <decode outputs, thresholds, resize back, copies to the CPU> | | |
| Service post-processing | <encoding, response building> | | |

## Model
- Architecture: <network type and structure: backbone, neck, heads, blocks, parameter count; what the task is>
- Inputs: <name, shape, dtype, value range, layout, color order for each input>
- Outputs: <name, shape, dtype, meaning for each output>
- Runtime: <framework or engine, precision, how it is built and loaded, workers per GPU>
- Notable operations: <attention, normalization, plugins, custom ops, dynamic shapes, anything that sets speed or precision>

## Top kernels
<From Nsight Systems on the tree named in "Measured on": the five kernels with the largest share of GPU time per request, each with its share, its time, and the layer and operation it belongs to; the command and the raw report's path in lx-work/local/profiles/.>

## Hot spots
- <path:function>: <why it matters, with the profile evidence>

## Notes
- <Surprises such as global state, hidden syncs, lazy initialization.>
