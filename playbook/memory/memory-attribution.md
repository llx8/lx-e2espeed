---
tier: on-demand
read_when: profiling shows the symptom below
---
# Attributing peak GPU memory

- Category: memory
- Applies to: PyTorch services and pipelines, with or without TensorRT engines, whose peak memory is the target or an OOM risk

## Symptom
Peak memory too high or an OOM, and per-stage deltas of `max_memory_allocated()` read zero after the tallest stage (it is monotonic over the process).

## Technique
- Each stage's own height: synchronize, read `memory_allocated()` as the floor, `reset_peak_memory_stats()`, run the stage, synchronize, read the peak; peak minus floor is its transient. Prove an attribution by predicting the peak without the tallest stage, then measuring it.
- Look at the floor first: resident weights under a high floor make any transient small money.
- `nvidia-smi` per process against PyTorch's allocated and reserved: the gap is memory outside PyTorch (CUDA context, TensorRT workspace).
- A workspace pool shared by several engines: instrument the pool's reserve call with its callers' full paths (basenames collide), check the pool against the largest engine's need, and read each engine's `get_tensor_profile_shape` before proposing a smaller pool.
- An allocator snapshot (`torch.cuda.memory._record_memory_history()`, `_dump_snapshot()`, private APIs, diagnostics only) at the OOM or the peak; report its path, input dimensions, repetition, and whether it was an OOM. Reserved growing while allocated stays flat is fragmentation; a leak is final allocated drifting up across requests.
- Large inputs: several real aspect ratios at the maximum long edge in PROJECT.md, three repeats per shape in one warm process, recording latency, peak and final memory, and tile counts; `gc.collect()` and one `empty_cache()` only at the request boundary. Fail on OOM, a non-finite or missing output, a wrong shape, a seam, or monotonic growth; rerun the matrix after a fix.
- Mixed workloads: the cartesian product of production's shapes and modes, shuffled with a fixed seed in one process, every mode warmed first, the peak reset per request, no cleanup; report the shape set, mode set, request count, and seed.

## Effect
Names the stage, engine, or mode that sets the peak, so a fix is aimed where memory goes; one diffusion service's real saving was 844 MiB where a single-stage reading predicted 3.4 GiB.

## Side effects
The extra syncs and snapshots change timing: never take latency from these runs.

## Does not apply
- One shape in one mode when production mixes them: `expandable_segments` measured 116 MiB on one shape and 2692 MiB over 7 shapes in 6 modes interleaved.

## Evidence
Method from the retired memory skill, absorbed into the rules on 2026-10-13 (937dc71), moved here from rules/benchmarking.md on 2026-10-20.
- ✓ before 2026-10 (from that skill): diffusion service, five stages peaking at 22.3 to 25.2 GiB over a 19.5 GiB floor of resident weights (a plateau, not one spike): removing the tallest saved 844 MiB, not the 3.4 GiB predicted; mode switches alone cost 864 MiB, and the floor depended on the previous request's LoRA.
