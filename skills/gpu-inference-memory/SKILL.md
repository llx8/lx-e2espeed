---
name: gpu-inference-memory
description: "Cut and validate peak GPU memory: large or maximum inputs, tiling, ROIs, VAE, OOM, fragmentation, memory snapshots."
---

# Optimize Peak Inference Memory

Optimize the live peak, not only the final allocation count.

## Workflow

1. Build a budget for weights, persistent caches, engine workspace, activations, outputs, and temporary tensors.
2. Finish the main E2E latency and dataflow work before running maximum-image stress tests.
3. Determine the maximum input the service contract allows (for example a long edge of 13000 px). Do not invent a maximum the service does not support.
4. Run [references/large-image-validation.md](references/large-image-validation.md) at that maximum with multiple real aspect ratios and repeated complete E2E requests.
5. Enable CUDA memory history before the suspect run and dump a snapshot on OOM or unexplained peak growth.
6. Tell the user where the snapshot was saved, what input and iteration produced it, and that the artifact is required for allocation-lifetime analysis.
7. Use [references/snapshot-analysis.md](references/snapshot-analysis.md) to identify allocations live at the peak.
8. Apply bounded-memory processing to the responsible stage, then repeat the complete maximum-image matrix.

Read [references/patterns.md](references/patterns.md) for supported patterns. Use `scripts/cuda_memory_snapshot.py` to wrap a real E2E call and `scripts/prepare_large_image_matrix.py` to prepare representative long-edge inputs.

## Solution routing

- For large-image preprocessing and postprocessing, first remove full-resolution copies, then use ordinary tiles for local operations or overlap tiles when neighborhoods, filters, restoration, or blending require context across boundaries.
- For masks, ROI blend, paste-back, resize, morphology, color processing, and accumulation, keep only tile metadata and process one bounded region at a time.
- For a large image entering a VAE, use Diffusers VAE tiling first. If profiling proves the convolutional VAE is still material and its tile contract is stable, combine Diffusers-compatible tiling semantics with the CNN TensorRT static encoder/decoder tile implementation.
- Do not tile a stage that is not live at the memory peak.

## Guardrails

- Do not use `empty_cache()` as a substitute for fixing live tensor lifetime.
- Call `empty_cache()` only after the full E2E request outputs and temporary references are released; never use it between stages to make an invalid design appear stable.
- Do not pre-create every full-size tile, mask, or accumulation tensor.
- Do not solve memory pressure by reducing precision or service resolution. Preserve the production contract and pass the complete project accuracy suite.
- Do not assume a path that fits one image will remain stable under maximum ROI or engine workspace.
