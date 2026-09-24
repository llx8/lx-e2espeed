# Mixed-Workload Memory Validation

## Trigger

Run this whenever the service switches between modes/effects **and** accepts a
range of input resolutions. Single-variable tests systematically mislead on such
a service: they under-measure fragmentation, and they cannot see the cost of
mode switching at all.

## Why single-variable tests mislead

Measured on one production pipeline (RTX 5090, six effects, 13000 max long
edge). The same `expandable_segments:True` allocator setting, evaluated four
ways:

| Test workload | Reserved peak, off -> on | Apparent value |
| --- | --- | ---: |
| One common shape, one effect | 27388 -> 27272 MiB | 116 MiB (noise) |
| One maximum-size square, one effect | 28014 -> 26168 MiB | 1846 MiB |
| Five long-edge shapes, one effect | 25556 -> 24248 MiB | 1308 MiB |
| **Seven shapes x six effects, interleaved** | **27804 -> 25112 MiB** | **2692 MiB** |

The first row led to the setting being removed as worthless. The last row is the
production workload. A 23x error in the estimate came purely from test shape.

The same run exposed two effects invisible to single-variable tests:

- **Mode switching raised the peak by 864 MiB.** Worst case under one effect
  across many shapes was 24248 MiB; interleaving effects reached 25112 MiB.
- **The resident floor moves with the previous request.** Per-mode LoRA state
  stayed resident until another mode replaced it: 656 MiB after one effect,
  404 MiB after another, 0 after the rest. The floor a request starts from
  therefore depends on the request before it, and the worst path (largest
  residual, then largest input) never occurs in a single-effect test.

## Protocol

1. Build the shape set from real images: several aspect ratios at the maximum
   long edge (see `prepare_large_image_matrix.py`), plus common production
   sizes. Distinct **sizes** are what create allocator segment mismatch;
   aspect ratio only matters because it produces distinct sizes.
2. Take the cartesian product of shapes and modes, shuffle with a fixed seed,
   and run it in **one process**. Interleaving is the point — consecutive
   identical shapes let the allocator reuse the previous segment and hide the
   fragmentation the service actually suffers.
3. Warm every mode once before measuring, so engine/LoRA/autotune costs do not
   land in the first measured request.
4. Reset peak statistics immediately before each request.
5. **Do not insert extra `gc.collect()` / `empty_cache()` between requests.**
   Keep only the cleanup the production request path performs. Added cleanup
   hides exactly the cross-request residue this test exists to find. (This
   differs from `large-image-validation.md`, which isolates one shape on
   purpose.)
6. Record per request: latency, peak allocated, peak reserved, final allocated,
   final reserved, and the validation result.

## Reading the result

Separate the two drift signals; they have different causes and different fixes:

- **Final allocated drifting up** = live tensors retained across requests, i.e.
  a leak. Look for containers, closures, or module attributes holding request
  tensors.
- **Final reserved drifting up while allocated is flat** = fragmentation
  accumulating. This is what an expandable-segment allocator addresses; it is
  not a leak and `empty_cache()` only masks it.

Bounded oscillation in final allocated is not a leak when it tracks which mode
ran last — verify by correlating the value with the preceding mode before
investigating further.

Fail on: any OOM, non-finite output, missing output, wrong output shape, or
monotonically growing final allocated. Capture a snapshot per
`snapshot-analysis.md` on OOM or when the peak leaves insufficient margin.

## Reporting

State the shape set, the mode set, the request count, and the interleaving seed.
A peak number without the workload that produced it is not comparable to
anything, and — as the table above shows — is the single easiest way to reach a
confidently wrong conclusion.
