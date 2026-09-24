# Attributing a Memory Peak to a Stage

## Contents
- Do not attribute from the global high-water mark alone
- Measure absolute per-stage height
- Expect a plateau, not a spike
- Attributing a shared TensorRT workspace pool

Two questions look alike and are not:

- *Which stage pushed the global high-water mark?* — answered by watching
  `max_memory_allocated()` grow across stages.
- *How tall is each stage on its own?* — requires resetting peak statistics per
  stage.

Answering the first and acting as if it answered the second produces wrong
optimization targets. This reference exists because that mistake was made.

## Do not attribute from the global high-water mark alone

`max_memory_allocated()` is monotonic. Wrap sequential stages and record the
delta and the peak is tempting:

```python
hwm_in = torch.cuda.max_memory_allocated()
run_stage()
delta = torch.cuda.max_memory_allocated() - hwm_in     # "this stage's cost"
```

`delta` is only *how much this stage exceeded everything before it*. Every stage
after the tallest one reports `delta == 0` — whether it is 5% shorter or 90%
shorter. A table of zeros reads as "only the first stage matters" when the truth
may be that four stages are within a few hundred MiB of each other.

That misreading led to a prediction of -3.4 GiB from optimizing the apparent
single peak owner. Measured result: **-844 MiB**, because the next stage down
was only ~850 MiB lower and immediately became the new ceiling.

## Measure absolute per-stage height

For stages that are **sequential and non-overlapping**, reset per stage:

```python
torch.cuda.synchronize()
floor = torch.cuda.memory_allocated()      # what earlier stages still hold
torch.cuda.reset_peak_memory_stats()
run_stage()
torch.cuda.synchronize()
absolute_peak = torch.cuda.max_memory_allocated()   # true ceiling this stage reached
own_transient = absolute_peak - floor               # this stage's own working set
```

The global peak is then `max(absolute_peak)` over stages. Resetting is safe only
because the stages do not overlap; do not reset inside a stage that is nested in
another instrumented stage — instrument one level at a time, or make the leaf
stages the only instrumented ones.

Validate the instrumentation before trusting it: predict the peak after removing
the tallest stage (it should equal the second-tallest absolute peak), then make
that change and confirm. On the pipeline this reference came from, the table
predicted 24903 MiB and the change measured 24903 MiB.

## Expect a plateau, not a spike

A mature pipeline usually has several stages sized by the same input, so their
peaks cluster:

| Stage | Absolute peak | Own transient |
| --- | ---: | ---: |
| VAE encode (full frame) | 25756 MiB | 6168 MiB |
| Post-processing | 24903 MiB | 4364 MiB |
| Transformer step | 24290 MiB | 4677 MiB |
| VAE decode (streamed) | 24230 MiB | 4617 MiB |
| CNN restore, full size | 22801 MiB | 3230 MiB |

Shaving the tallest column buys the gap to the next one, not its full height.
Here, fixing all four top stages would return 2.9 GiB, while the 19.5 GiB floor
under them was 76% of the peak. **Check the floor before optimizing columns:**
every GiB removed from the floor removes a GiB from every column at once, with
no dependence on which stage is currently tallest.

## Attributing a shared TensorRT workspace pool

A pool sized `max(engine device_memory_size)` charges every request for the
largest engine in the process, including requests that never execute it. To find
the engine responsible, instrument the pool's reserve entry point and record the
**full path** of the caller:

```python
import traceback
for frame in reversed(traceback.extract_stack()[:-1]):
    if "shared_runtime" not in frame.filename:
        caller = f"{frame.filename}:{frame.lineno}"   # full path, not basename
        break
```

Two independent pitfalls, both hit on the same investigation:

- **Basename collides.** Two vendored subsystems each shipped a `deep_engine.py`.
  Reporting `deep_engine.py:44` attributed the pool to the wrong subsystem
  twice, and two rounds of changes were made against the wrong target before
  full-path logging identified the real one.
- **Not every engine goes through the primary loader.** Wrapping only the
  project's own TensorRT module class missed vendored loaders that call the pool
  directly. Instrument the **pool**, not the loader.

Cross-check the total: if the pool is larger than the largest engine you
enumerated, an allocator you have not instrumented is reserving from it. That
discrepancy is the signal — chase it rather than explaining it away.

Before proposing to shrink an engine's workspace, read its optimization profile.
A workspace can be inherent to the tile size rather than an over-wide profile:

```python
mn, opt, mx = engine.get_tensor_profile_shape(name, profile_index)
```

If `max` already matches the production tile, the workspace cannot be reduced by
narrowing the profile — only by rebuilding at a smaller tile, which is a
different change with its own quality and latency consequences.
