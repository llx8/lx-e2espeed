---
name: gpu-inference-diffusion
description: Load user-provided Nunchaku quantized weights for Flux/Kontext and convert LoRA or ComfyUI assets to diffusers.
---

# Integrate Nunchaku Diffusion Inference

Keep this skill narrow: validate user-supplied quantized weights, normalize source formats to diffusers contracts, and integrate Nunchaku through maintained upstream APIs.

## Production backend

Whether Nunchaku is the production backend is decided per project. Once it is selected, the project must expose an architecture-matched, already-quantized checkpoint where it keeps its models. A missing checkpoint, unsupported architecture, rank mismatch, or incompatible LoRA must fail before inference; the production path must not continue with a full-precision diffusers transformer. Keep full precision only as an explicitly selected offline accuracy reference.

## Required user input

Require the user to provide the already-quantized Nunchaku transformer weights or an explicit accessible local path. Do not start custom Nunchaku quantization, guess a checkpoint, or silently substitute a different base model.

Also identify:

- exact base model and architecture variant;
- weight precision/quantization variant and Nunchaku version;
- diffusers pipeline type and local base-model components;
- LoRA files, source format, rank, strength, and switching requirements;
- the original xFlux or ComfyUI reference workflow needed for output comparison.

## Workflow

1. Inspect weight metadata, filenames, state-dict keys, architecture, and required Nunchaku loader before allocating the full model.
2. Confirm the base model, VAE, text encoders, scheduler, image dimensions, steps, CFG, seed, conditioning, and preprocessing used by the reference path.
3. If assets already follow the expected diffusers layout and key conventions, load them without conversion.
4. If LoRA uses xFlux/XLabs/ai-toolkit keys, convert it with `scripts/convert_xflux_lora_to_diffusers.py`.
5. If Kontext/Fill LoRA uses raw processor keys, convert it with `scripts/convert_fill_lora.py`.
6. If inference is defined by a ComfyUI workflow, map the workflow components and settings to the matching diffusers pipeline first. Do not treat workflow JSON as a model conversion.
7. Validate the explicitly selected diffusers eager reference against the original xFlux or ComfyUI result before inserting Nunchaku.
8. Load the user-provided quantized transformer through the upstream Nunchaku API, select it as the production backend if the project decided so, and validate again.
9. When SageAttention is selected for a current Flux/Flux2 Nunchaku build, use the native process-local backend API described in the reference. Enable it after constructing the diffusers pipeline because pipeline setup may reset the backend.
10. Use `scripts/nunchaku_v2_lora_switch_example.py` only when multiple compatible LoRAs must switch without rebuilding the pipeline.
11. Route large-image VAE memory work to the `gpu-inference-memory` skill and a measured convolutional VAE hotspot to `gpu-inference-cnn-tensorrt`.

## Guardrails

- Do not claim conversion success merely because weights load. Compare generated output with aligned inputs and settings.
- Do not fuse, compress, or change LoRA rank without explicit approval and visual/metric validation.
- Prefer native Nunchaku attention support. For current Flux/Flux2 builds, use `set_flux_attention_backend`; do not restore a local SageAttention monkey patch.
- Do not enable `torch.compile` from this skill.

Read [references/nunchaku-weights-and-formats.md](references/nunchaku-weights-and-formats.md) before loading or converting assets.
