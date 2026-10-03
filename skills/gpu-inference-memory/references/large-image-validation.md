# Maximum-input Validation

## Trigger

Run this only after the main E2E optimization is stable, at the maximum input the product contract permits. The examples below use a 13000 px long edge; substitute the project's own maximum.

## Input matrix

Select multiple real test images with materially different non-square aspect ratios, including landscape, portrait, and wide or tall content relevant to production. Preserve each aspect ratio and resize it so:

```text
max(width, height) == 13000
min(width, height) < 13000
```

Use `scripts/prepare_large_image_matrix.py`. Do not stretch one image into arbitrary aspect ratios, because that changes masks, detections, ROIs, and stage behavior.

## Repeated complete E2E test

For every generated shape:

1. Start from the real service/demo entry point and include decode, preprocessing, every model, postprocessing, blend/paste-back, output transfer, and encode/save when those belong to the request.
2. Reset peak statistics immediately before the request.
3. Run the same shape at least three times in one process after required warmup. Increase repetitions when production reuses the process for many requests.
4. Record E2E latency, peak allocated, peak reserved, final allocated, final reserved, tile/ROI counts, and whether output validation passed.
5. Release request outputs and temporary references, run `gc.collect()`, then call `torch.cuda.empty_cache()` once at the request boundary.
6. Record allocated/reserved memory again and continue to the next repetition.
7. Fail the test on any CUDA OOM, non-finite output, missing output, incorrect shape, visible tile seam, or monotonic live-memory growth.

`empty_cache()` cannot free live tensors. A test that OOMs before the request boundary is still a failure and must not be repaired by inserting cache clears between stages.

## Snapshot condition

Capture a snapshot when an OOM occurs, peak memory leaves insufficient safety margin, reserved memory grows unexpectedly, or final allocated memory does not return near the established warm level. Tell the user the snapshot path, image dimensions, aspect ratio, repetition index, and failing stage if known.

After a fix, rerun the entire matrix rather than only the previously failing image.
