---
tier: once
read_when: before writing or changing a bench or verify tool
---
# Bench and verify tools

Every bench or verify tool in lx-work/tools/ follows these rules.
Why: most speedups that later vanish were measurement errors, not real changes.

## Timing
- Report separately: process start and model loading, engine build or compilation, the first call, warmup, and the steady state. Cold start is its own number, never mixed into latency.
- Warm up until timings stabilize. Early iterations include compilation, autotuning, allocation, and clock ramp-up.
- Time the end-to-end path within the measurement boundary written in PROJECT.md (for example decoded input to final output, excluding file I/O) with a wall clock. Use CUDA events for GPU regions, and synchronize only at measurement boundaries, never inside production stages.
- Report median, P90, and P99, and the spread across repeats. Record peak allocated and reserved memory (reset the peak statistics before each run), plus memory outside PyTorch, such as TensorRT, from `nvidia-smi` per process.
- Give baseline and candidate the same inputs, shapes, and process conditions, and alternate them in one session; `exp.sh export best` or `baseline` gives a runnable copy of the other side. Run them in separate processes when they share library names or global state; never measure one in a process that inherited the other's allocator or cache state.
- Give the bench an A/B mode that alternates baseline and candidate for at least three rounds, one fresh process per scenario and side, and reports each difference against the noise floor. Scenarios sharing a process shift each other: state left by one (allocator, caches) moved the next by several milliseconds in one project. Confirm any unexpected gain with the scenario order changed.
- Keep one bench run short enough to repeat often; agree on a time budget at onboarding. A run that takes over twice its usual time is killed and counted as a failure.

## Inputs
- Use real inputs at production resolution. Never downsample to save time, and never fabricate cases that production does not produce, such as a single image cut into artificial regions.
- Cover the common shape, the maximum supported input, tile or ROI boundaries, and data-dependent cases such as empty masks, each reported separately.
- Record which environment produced each number. Numbers from different environments or hardware are not comparable and must not be subtracted.

## Noise floor
Run the bench at least three times, each in a fresh process, on the same unchanged code, with the same process layout the A/B mode uses. The spread of the medians is the noise floor.
Measure it at baseline and again whenever the machine, driver, or environment changes. A difference smaller than the noise floor is not a gain.

## Time breakdown
Give the bench a stage mode that times each step of the MAP.md path, synchronizing at every step boundary.
Use it only for the breakdown: the extra syncs change the end-to-end number, so headline numbers always come from the normal mode.
Wrap callables with timers rather than editing project code. Steps written inline cannot be wrapped: split that remainder with a function-level CPU profiler (cProfile, py-spy) and say which numbers came from which.

## GPU state
- Record SM clock, temperature, and power during the run: `nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv -lms 500`.
- Lock the SM clock for any A/B (`nvidia-smi -lgc <MHz>`, reset with `nvidia-smi -rgc`). If it cannot be locked, record `clocks.sm` for both sides and report the drift. A round without a clock reading does not count, and must never be reported as "no drift". Unlocked, the drift favors the faster candidate, which leaves more thermal and power headroom. On an RTX 5090 a lock binds only approximately (7.5 MHz steps, software power cap), so report the residual.
- After a GPU, driver, or environment change, re-measure the baseline in the same session; older absolute numbers are no longer comparable. On the same card, absolute milliseconds across rounds are comparable, and comparing them is often how drift gets caught.
- Nothing else runs on the GPU while measuring. Before blaming an OOM on a change, check `nvidia-smi` for other processes.

## Correctness
- Compare against the frozen reference outputs in baseline/, with the method, metrics, and tolerance in PROJECT.md, over the whole input matrix above. Fix seeds; if outputs are inherently nondeterministic, agree on a comparison metric with the user.
- Grade every result: L0 identical; L1 within the noise of rebuilding the same reference (for example an engine rebuilt from the same ONNX); L2 within reduced-precision noise; L3 larger. PROJECT.md sets the thresholds and the accepted level; the user decides.
- A "difference" is a difference in the final output of the real end-to-end path, stated with its scenario and input size, as magnitude (maximum error) and extent (share of values or pixels affected). How close one operator is to the exact math is "operator accuracy": a diagnostic, never reported as a difference. Never mix scenarios in one statement.
- For images, report MAE, RMSE, PSNR, maximum error, the share beyond the agreed threshold, and the share changed, against the pass criterion in PROJECT.md.
- A check must be able to fail: an exact-equality check can pass because an approximate branch never ran. Prove it once with a deliberately wrong input or setting.
- Compare at the final service output too: encoding such as JPEG can amplify a 1 to 3 level difference into a much larger one.
- Rebuilding an engine changes its outputs even from the same ONNX, so agree on a tolerance before any engine change, and compare every mixed-precision engine end to end: they can be silently wrong.

## Output
- Print a summary of at most 10 lines, with library logs sent to a file at the file-descriptor level (native libraries such as TensorRT write past `sys.stdout`; flush Python's buffers before switching): metric, median, P90, P99, spread, peak memory, change against baseline and best, difference level, verify pass or fail. Tool output costs context, so full data goes to files.
- Write full results as JSON, including the environment (GPU, driver, CUDA, framework and TensorRT versions, SM clock) and the project commit.
- A value that was not measured is "not recorded", never zero or an estimate.
