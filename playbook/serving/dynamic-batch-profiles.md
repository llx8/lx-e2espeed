---
tier: on-demand
read_when: profiling shows the symptom below
---
# One optimization profile per batch size

- Category: serving
- Applies to: TensorRT engines called with a varying number of same-size items per request (faces, crops, tiles)

## Symptom
A request with several items calls the engine once per item, so tail latency grows with the item count; or one dynamic profile with a large opt makes batch 1 slow.

## Technique
Export the ONNX with a dynamic batch dimension (fix reshape targets that hard-code 1). Build one static profile per batch size 1 to N (min = opt = max = b) in one engine and one execution context, so each batch runs tactics chosen for it. Capture a CUDA Graph per batch size at Init: serving only replays graphs and never switches profiles. Write a request's items into the rows of one N-row input buffer. Pick N where per-item throughput stops improving.

## Effect
P99 8.95 -> 5.46 ms with the median unchanged; 2 items 2.14 -> 1.89 ms, 9 or more 8.94 -> 5.37 ms; L1 (a rebuilt engine).

## Side effects
Memory grows with N (28 profiles: engine +258 MiB, process 1350 -> 1802 MiB); a longer build; Init captures N graphs.

## Does not apply
- One dynamic profile with opt at the largest batch: batch 1 ran 2.6x slower through large convolution tiles.
- Pipelining items over batch-1 slots with event waits gave -4.6% and was landed, then removed by the user: production's GPU also runs other work, where its behavior was uncertain.

## Evidence
- ✓ 2026-10: face classification CNN service, 512x512 crops, 1 to 28 faces per request, FP16, RTX 5090, TensorRT 10.16: median 1.80 -> 1.79 ms, P99 8.95 -> 5.46 ms, L1, no label change.
