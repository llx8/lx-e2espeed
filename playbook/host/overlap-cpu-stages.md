---
tier: on-demand
read_when: profiling shows the symptom below
---
# Overlap independent CPU stages

- Category: host
- Applies to: request paths with independent CPU-heavy stages that release the GIL (image decode or encode, zlib)

## Symptom
The time breakdown shows two independent CPU stages back to back, for example the request image's decode and the masks' decode, or two response encodes.

## Technique
Run one of them on a small thread pool created at Init and join before its result is used; keep the pool size fixed. Check that the library releases the GIL, or the overlap is empty.

## Effect
-5.16 ms (-9.9%) for two decodes, -0.50 ms (-2.4%) for two encodes; L0.

## Side effects
More threads per worker: with many workers the extra threads compete for cores.

## Does not apply
- A second CUDA stream for a small engine beside a large one: -0.51 ms, but the user took it back for one stream (simpler under several workers).

## Evidence
- ✓ 2026-10: two-model mask service, the request JPEG decoded on a one-thread pool while the PNG masks decode: -5.16 ms (-9.9%), L0.
- ✓ 2026-10: the same service, one PNG response mask encoded on a second thread: -0.50 ms (-2.4%), L0.
