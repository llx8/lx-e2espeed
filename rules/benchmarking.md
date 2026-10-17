---
tier: on-demand
read_when: before an A/B or noise-floor run, judging a difference level, or writing or changing a bench or verify tool
---
# Bench and verify tools

Every bench or verify tool in lx-work/tools/ follows these rules.
Why: most speedups that later vanish were measurement errors, not real changes.

## Timing
- Report separately: process start and model loading, engine build or compilation, the first call, warmup, and the steady state. Cold start is its own number, never mixed into latency.
- Warm up until timings stabilize. Early iterations include compilation, autotuning, allocation, and clock ramp-up.
- Time the end-to-end path within the measurement boundary written in PROJECT.md (for example decoded input to final output, excluding file I/O) with a wall clock. Use CUDA events for GPU regions, and synchronize only at measurement boundaries, never inside production stages.
- Report median, P90, and P99, and the spread across repeats. P99 needs at least 100 samples; with fewer, report the maximum instead. Record peak allocated and reserved memory (reset the peak statistics before each run), plus memory outside PyTorch, such as TensorRT, from `nvidia-smi` per process.
- Give the bench an A/B mode that alternates the project tree (best) and the candidate in one session, with the same inputs, shapes, and process conditions: at least three rounds, swapping which side runs first each round, one fresh process per scenario and side, each difference reported against the noise floor.
  Separate processes, because code sharing library names or global state cannot share one, and scenarios sharing a process shift each other: allocator and cache state left by one moved the next by several milliseconds in one project. To compare against another tree, pass `--project` with a copy from `exp.sh export baseline-commit` or `NNN`. Confirm any unexpected gain with the scenario order changed.
- Keep one bench run short enough to repeat often; agree on a time budget at onboarding (PROJECT.md, Metric), which WARMUP and REPEAT in the adapter fit. A run that takes over twice its usual time is killed and counted as a failure.

## Inputs
- Use real inputs at production resolution. Never downsample to save time, and never fabricate cases that production does not produce, such as a single image cut into artificial regions.
- Cover the common shape, the maximum supported input, tile or ROI boundaries, and data-dependent cases such as empty masks, each reported separately.
- Record which environment produced each number. Numbers from different environments or hardware are not comparable and must not be subtracted. One exception: when the user made another environment's speed the baseline (DECISIONS.md), compare against it on the same machine.
- A change against the baseline is descriptive, not a paired A/B: its numbers come from another session, and from another environment when the user chose one. Before a conclusion rests on it (a milestone, wrap-up), bench both sides in one session, alternating three runs each.

## Noise floor
Measure it as an A/A run of the A/B mode (`bench --ab best`): the same code on both sides, the same process layout, at least three rounds. The largest difference per scenario is that scenario's noise floor; it goes to `NOISE_MS` in the adapter, which the A/B reads, and to STATUS.md.
Measure it on the baseline commit during onboarding, again after gains that change latency by a large factor (noise shrinks with it), and whenever the machine, driver, or environment changes. A difference smaller than the noise floor is not a gain.

## Time breakdown
Give the bench a stage mode that times each step of the MAP.md path, synchronizing at every step boundary.
Use it only for the breakdown: the extra syncs change the end-to-end number, so headline numbers always come from the normal mode.
Wrap callables with timers rather than editing project code. Steps written inline cannot be wrapped: split that remainder with a function-level CPU profiler (cProfile, py-spy) and say which numbers came from which.

## GPU state
- Record SM clock, temperature, and power during the run: `nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv -lms 500`.
- Keep the SM clock locked (`nvidia-smi -lgc <MHz>`) at all times, so every session measures at the same clock. Reset it (`nvidia-smi -rgc`) only in a special case and with the user's okay, for example when others need the GPU.
  A reboot or driver reload drops the lock, and `clocks.sm` cannot always show it (a loaded GPU may run below its lock under the power cap): before an A/B, lock again at the same value if the machine rebooted (`uptime`) or the driver reloaded since it was set. If it cannot be locked, record `clocks.sm` for both sides and report the drift. A round without a clock reading does not count, and must never be reported as "no drift".
  Unlocked, the drift favors the faster candidate, which leaves more thermal and power headroom. On an RTX 5090 a lock binds only approximately (7.5 MHz steps, software power cap), so report the residual.
- After a GPU, driver, or environment change, bench best and the baseline again in the same session (the baseline in its own environment); older absolute numbers are no longer comparable. On the same card, absolute milliseconds across rounds are comparable, and comparing them is often how drift gets caught.
- Nothing else runs on the GPU while measuring. Before blaming an OOM on a change, check `nvidia-smi` for other processes, and reconcile its per-process number with PyTorch's allocated and reserved: the gap is memory outside PyTorch (CUDA context, TensorRT workspace).

## Memory
- `max_memory_allocated()` is monotonic over the process, so per-stage deltas read zero after the tallest stage. Measure each stage's own height: synchronize, read `memory_allocated()` as the floor, `reset_peak_memory_stats()`, run the stage, synchronize, read the peak; peak minus floor is the stage's transient. Prove an attribution by predicting the peak without the tallest stage, then measuring it.
- Expect a plateau, not one spike: in one diffusion service five stages peaked between 22.3 and 25.2 GiB over a 19.5 GiB floor of resident weights, so removing the tallest saved 844 MiB where 3.4 GiB was predicted. Look at the floor first; a transient under a high floor is small money.
- A workspace pool shared by several engines: instrument the pool's reserve call with its callers' full paths (file basenames collide), cross-check the pool against the largest engine's need, and read each engine's `get_tensor_profile_shape` before proposing a smaller pool.
- An allocator snapshot (`torch.cuda.memory._record_memory_history()` and `_dump_snapshot()`, private APIs, diagnostics only) is taken at the OOM or at the peak; report its path, the input dimensions, the repetition, and whether it was an OOM. Reserved memory growing while allocated stays flat is fragmentation; a leak is final allocated memory drifting up across requests.
- Large inputs: several real aspect ratios resized to the maximum long edge in PROJECT.md, at least three repeats per shape in one warm process, recording latency, peak and final allocated and reserved memory, and tile counts. `gc.collect()` and one `empty_cache()` only at the request boundary, never between stages. Fail on OOM, a non-finite or missing output, a wrong shape, a seam, or monotonic growth, and rerun the whole matrix after a fix.
- Mixed workloads: one shape in one mode misleads when production mixes them. The effect of `expandable_segments` measured 116 MiB on one shape and 2692 MiB over 7 shapes in 6 modes interleaved; switching modes alone cost 864 MiB, and the resident floor depended on which LoRA the previous request had loaded.
  Run the cartesian product of shapes and modes shuffled with a fixed seed in one process, warm every mode first, reset the peak per request, add no cleanup, and report the shape set, mode set, request count, and seed.

## Correctness
- Compare against the frozen reference outputs in baseline/, with the method, metrics, and tolerance in PROJECT.md, over the whole input matrix above. That shows whether outputs changed; how good they are is measured against the quality baseline (`rules/quality.md`). Fix seeds; if outputs are inherently nondeterministic, agree on a comparison metric with the user.
- Grade every result: L0 identical; L1 within the noise of rebuilding the same reference (for example an engine rebuilt from the same ONNX); L2 within reduced-precision noise; L3 larger. PROJECT.md sets the thresholds and the accepted level; the user decides. The user may set none: then every result that is not L0 goes to the user with verify's numbers.
- A "difference" is a difference in the final output of the real end-to-end path, stated with its scenario and input size, as magnitude (maximum error) and extent (share of values or pixels affected). How close one operator is to the exact math is "operator accuracy": a diagnostic, never reported as a difference. Never mix scenarios in one statement.
- For images, report MAE, RMSE, PSNR, maximum error, the share beyond the agreed threshold, and the share changed, against the pass criterion in PROJECT.md.
- A check must be able to fail: an exact-equality check can pass because an approximate branch never ran. Prove it once with a deliberately wrong input or setting.
- When a change adds or alters a branch the scenario inputs never reach, add a targeted test that reaches it, derived from real inputs, and keep it with the experiment. It compares against the previous code, so after landing run it against `exp.sh export` of the previous experiment.
- Compare at the final service output too: encoding such as JPEG can amplify a 1 to 3 level difference into a much larger one.
- Rebuilding an engine changes its outputs even from the same ONNX, so agree on a tolerance before any engine change, and compare every mixed-precision engine end to end: they can be silently wrong.
- When an engine differs beyond the agreed level, find the first failing boundary and bisect inside it, as `playbook/compute/engine-difference-triage.md` describes; a global FP32 switch is a diagnostic, not the repair.

## Output
- Print a summary of at most 10 lines, with library logs sent to a file at the file-descriptor level (native libraries such as TensorRT write past `sys.stdout`; flush Python's buffers before switching): metric, median, P90, P99, spread, peak memory, and the baseline's numbers beside them with the change against the baseline (descriptive).
  An A/B prints three sets per scenario, baseline, best (before), and the candidate (after), and the candidate's change against both, judged against the noise floor only for best. Verify prints the difference level and pass or fail. Tool output costs context, so full data goes to files.
- Write full results as JSON, including the environment (GPU, driver, CUDA, framework and TensorRT versions, SM clock) and the project commit. An A/B writes every round of both sides to its own `--out` file; its per-round files are scratch that the next A/B overwrites.
- A value that was not measured is "not recorded", never zero or an estimate.
