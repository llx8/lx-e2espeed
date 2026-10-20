---
tier: on-demand
read_when: writing or changing a bench, verify, or other measuring tool, in these templates or in lx-work/tools/
---
# How the measuring tools are built

The generic tools here implement `rules/benchmarking.md`; a project tool that measures follows the same design. A general fix found in a project comes back here at wrap-up.

## Bench
- Report separately: process start and model loading, engine build or compilation, the first call, warmup, and the steady state. Wall clock for the end-to-end path, CUDA events for GPU regions, syncs only at measurement boundaries, never inside production stages.
- Reset the peak memory statistics before each run; record peak allocated and reserved, and memory outside PyTorch (TensorRT) from `nvidia-smi` per process.
- A/B mode: one fresh process per scenario and side, because code sharing library names or global state cannot share a process, and scenarios sharing one shift each other (allocator and cache state moved the next scenario by several milliseconds in one project).
- One run stays short enough to repeat often: WARMUP and REPEAT in the adapter fit the time budget in PROJECT.md (Metric).
- Sample SM clock, temperature, and power during the run (`nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv -lms 500`). On one card, absolute milliseconds across rounds are comparable, which is often how drift gets caught.
- Stage mode wraps callables with timers rather than editing project code, and leaves no remainder: the engine call and each layer's entry function are wrapped too, a wrapped function counts only its own lines (nested stages subtracted), so stages add up to the total and "other" is the adapter's call alone. A native object whose method refuses a wrapper gets the pass-through proxy.

## Output
- A summary of at most 10 lines; library logs go to a file at the file-descriptor level (native libraries such as TensorRT write past `sys.stdout`; flush Python's buffers before switching). Tool output costs context, so full data goes to files.
- The summary: metric, median, P90, P99, spread, peak memory, and the baseline's numbers beside them with the change against the baseline (descriptive). An A/B prints three sets per scenario, baseline, best (before), and the candidate (after), with the candidate's change against both, judged against the noise floor only for best. Verify prints the difference level and pass or fail.
- Full results as JSON, with the environment (GPU, driver, CUDA, framework and TensorRT versions, SM clock) and the project commit. An A/B writes every round of both sides to its `--out` file; its per-round files are scratch the next A/B overwrites.
