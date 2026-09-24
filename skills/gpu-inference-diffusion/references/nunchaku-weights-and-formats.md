# Nunchaku Weights and Format Preparation

## Already-quantized weight gate

Ask the user for the quantized Nunchaku weight path and expected base-model identity. Verify that files exist and that their architecture and quantization variant match the installed Nunchaku loader. Custom quantization is outside this skill.

Do not combine a quantized transformer with an unrelated VAE, text encoder, scheduler, or pipeline class merely because tensor shapes happen to load.

Whether Nunchaku becomes the production transformer backend is a per-project decision. When it is selected, a full-precision transformer may be loaded only by an explicit offline reference or validation command, never as an automatic runtime fallback.

Expose validated deployment assets where the project keeps its models, resolved through its own path convention; never hardcode a source project's path or a remote weight store's absolute path. If the project deploys weights to a remote store, verify file identity there before handoff.

## diffusers-native assets

Inspect keys before conversion. Preserve already-compatible diffusers weights and LoRAs. Record the base model, adapter names, ranks, strengths, and expected target modules.

## Native SageAttention backend

For an installed Nunchaku version that exports the Flux backend selector, use the maintained API instead of patching attention classes or block forwards:

```python
import sageattention  # Validate the optional extension before selecting it.
from nunchaku.models.attention_processors.flux import set_flux_attention_backend

# Call after constructing the diffusers pipeline; pipeline setup may reset the
# process-local backend selected during transformer loading.
set_flux_attention_backend("sage", tensor_layout="HND")
```

Treat a requested Sage backend as fail-fast: an import or backend-selection failure must not silently continue with SDPA. Record the Nunchaku and SageAttention versions, selected layout, and transformer block count. Re-check the installed API rather than assuming this module path for older Nunchaku releases.

## xFlux, XLabs, and ai-toolkit LoRA

These formats may store Flux block and processor keys differently from diffusers. Use `scripts/convert_xflux_lora_to_diffusers.py` only after the detected keys match the converter's supported patterns. The converter must fail on unconsumed keys.

After conversion, check key count, target-module coverage, A/B tensor shapes, rank distribution, dtype, and generated output against the original xFlux path.

## Kontext and Fill raw LoRA

Use `scripts/convert_fill_lora.py` for supported raw processor key layouts. Rank reduction/SVD mode changes the adapter and requires explicit user approval plus output validation; it is not part of ordinary format normalization.

## ComfyUI reference path

Treat the ComfyUI workflow as the behavioral specification. Extract and align:

- checkpoint and component files;
- pipeline/task type;
- scheduler/sampler, steps, CFG, denoise strength, and seed;
- prompt/text-encoder behavior;
- image resize/crop, mask, reference image, ControlNet/adapter inputs;
- LoRA order, strengths, and target model;
- VAE encode/decode and final image processing.

Construct the corresponding diffusers pipeline and validate it before replacing the transformer with Nunchaku. Cache or acceleration differences are irrelevant until the output contract is aligned.

## Validation ladder

1. Original xFlux or ComfyUI reference.
2. Native diffusers eager path using compatible unquantized components.
3. diffusers path with converted LoRA, if conversion was required.
4. Nunchaku path with the user-provided quantized transformer.
5. Nunchaku dynamic LoRA switching, if required.

At each boundary, freeze inputs and settings and retain representative images. Stop at the first divergent boundary.
