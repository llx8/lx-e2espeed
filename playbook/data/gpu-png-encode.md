---
tier: on-demand
read_when: profiling shows the symptom below
---
# Byte-exact GPU PNG encode

- Category: data
- Applies to: services that return masks or images as PNG made by `cv2.imencode`, where the response bytes must stay identical

## Symptom
PNG encoding of the response masks is one of the largest stages: libpng's filtering and zlib on the CPU, one mask after another.

## Technique
Do the per-pixel part on the GPU: the filter choice and filtered rows, the run lengths, where each deflate block starts, and the Adler-32, then compress each block with the same zlib settings on a few CPU threads and join the bit streams. Write the GPU part as a few kernels (Triton), largest mask first. Check at Init that the bytes equal `cv2.imencode` on test masks.

## Effect
-2.23 ms (-18.6%) for the first version, -1.61 ms (-16.6%) more with the kernels; response bytes identical.

## Side effects
Ties the output to zlib's and libpng's exact choices: a library update can change the reference bytes; a checker against cv2 must run on every image build.

## Does not apply
- When byte-identical responses are not required, a GPU deflate library is simpler (not tried).

## Evidence
- ✓ 2026-10: two-model mask service, two PNG masks per response, RTX 5090: -2.23 ms (-18.6%), response bytes L0.
- ✓ 2026-10: the same service, the GPU part as three kernels: -1.61 ms (-16.6%), response bytes L0.
