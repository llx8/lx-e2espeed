---
tier: on-demand
read_when: before an A/B or noise-floor run, or judging a difference level
---
# Measuring speed and differences

Why: most speedups that later vanish were measurement errors, not real changes.
The bench and verify templates already implement what follows; changing a tool follows `templates/tools/DESIGN.md`. Memory attribution: card `playbook/memory/memory-attribution.md`.

## Timing
- Cold start (process start, model loading, engine build, first call) is its own number, never mixed into latency. Warm up until timings stabilize.
- Time the end-to-end path within the boundary in PROJECT.md. Report median, P90, P99 (P99 needs 100 samples; with fewer, the maximum), the spread across repeats, and peak memory, PyTorch's and outside it (`nvidia-smi` per process).
- A change is judged only by the A/B: best and the candidate alternated in one session, at least three rounds, the first side swapped each round, one fresh process per scenario and side, each difference against the noise floor. Another tree goes in with `--project` and a copy from `exp.sh export baseline-commit` or `NNN`. Confirm an unexpected gain with the scenario order changed.
- A run over twice its usual time is killed and counted as a failure.

## Inputs
- Real inputs at production resolution; never downsampled, never cases production does not produce (an image cut into artificial regions).
- Cover the common shape, the maximum supported input, tile or ROI boundaries, and data-dependent cases such as empty masks, each reported separately.
- Record which environment produced each number. Numbers from different environments or hardware are never subtracted; the one exception is a baseline the user took from another environment (DECISIONS.md), compared on the same machine.
- A change against the baseline is descriptive, not a paired A/B. Before a conclusion rests on it (a milestone, wrap-up), bench both sides in one session, alternating three runs each.

## Noise floor
An A/A run of the A/B mode (`bench --ab best`), at least three rounds: the largest difference per scenario is its noise floor, written to `NOISE_MS` in the adapter and to STATUS.md.
Measure it on the baseline commit at onboarding, again after a gain that changes latency by a large factor, and after any machine, driver, or environment change. A difference inside it is not a gain.

## Time breakdown
The bench's stage mode times each step of the MAP.md path with a sync at every boundary; its extra syncs change the total, so headline numbers always come from the normal mode. Steps written inline are split with a CPU profiler (cProfile, py-spy), saying which numbers came from which. Stages are named `<layer>.<operation>` after the layers in `rules/reporting.md`, Stage times.

## GPU state
- Nothing else runs on the GPU while measuring; check `nvidia-smi` for other processes first, and before blaming an OOM on a change.
- Keep the SM clock locked (`nvidia-smi -lgc <MHz>`); reset it (`-rgc`) only with the user's okay. A reboot or driver reload drops the lock: before an A/B, lock again if `uptime` or the driver says so. Unlocked, the drift favors the faster side.
  If it cannot be locked, record `clocks.sm` on both sides and report the drift; a round without a clock reading does not count and is never reported as "no drift". On an RTX 5090 a lock binds only approximately (7.5 MHz steps, software power cap): report the residual.
- After a GPU, driver, or environment change, bench best and the baseline again in one session; older absolute numbers are not comparable.

## Correctness
- Compare with the frozen references in baseline/ over the whole input matrix, with the method, metrics, and tolerance in PROJECT.md. That shows whether outputs changed; how good they are is measured against the quality baseline (`rules/quality.md`). Fix seeds; inherently nondeterministic outputs get a metric agreed with the user.
- Grade every result: L0 identical; L1 within the noise of rebuilding the same reference (an engine rebuilt from the same ONNX); L2 within reduced-precision noise; L3 larger. PROJECT.md sets the thresholds and the accepted level; with none set, every result that is not L0 goes to the user with verify's numbers.
- A "difference" is in the final output of the real end-to-end path, stated with its scenario and input size, as magnitude (maximum error) and extent (share of values or pixels affected). How close one operator is to the exact math is "operator accuracy", a diagnostic, never reported as a difference. Never mix scenarios in one statement.
- Images: MAE, RMSE, PSNR, maximum error, the share beyond the agreed threshold, and the share changed.
- A check must be able to fail: prove it once with a deliberately wrong input or setting, since an equality check passes when an approximate branch never ran.
- A changed branch the scenarios never reach gets a targeted test from real inputs, kept with the experiment; after landing, run it against `exp.sh export` of the previous experiment.
- Compare at the final service output too: JPEG encoding can turn a 1 to 3 level difference into a much larger one.
- Rebuilding an engine changes its outputs even from the same ONNX: agree on a tolerance before any engine change, and compare every mixed-precision engine end to end. Beyond the agreed level, triage as card `engine-difference-triage` says; a global FP32 switch is a diagnostic, not the repair.

## Results
A value that was not measured is "not recorded", never zero or an estimate.
