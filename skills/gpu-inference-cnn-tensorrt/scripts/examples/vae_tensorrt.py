import copy
import gc
import math
import os
from types import MethodType
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.utils import deprecate

from shared_trt_runtime import load_engine_from_pytorch


VAE_TRT_ENV = "GPU_INFERENCE_VAE_TRT"
VAE_TRT_FORCE_REBUILD_ENV = "GPU_INFERENCE_VAE_TRT_FORCE_REBUILD"
VAE_TRT_SKIP_SELF_CHECK_ENV = "GPU_INFERENCE_VAE_TRT_SKIP_SELF_CHECK"
VAE_TRT_MAX_BATCH = 1


def _env_flag(name: str, default: bool) -> bool:
    return bool(int(os.environ.get(name, "1" if default else "0")))


def _release_torch_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_module_device_and_dtype(module: nn.Module) -> Tuple[torch.device, torch.dtype]:
    try:
        param = next(module.parameters())
        return param.device, param.dtype
    except StopIteration:
        for buffer in module.buffers():
            return buffer.device, buffer.dtype
    raise RuntimeError("Unable to infer device/dtype from module.")


def _maybe_compile_module(module: nn.Module, label: str, mode: str = "default") -> Tuple[nn.Module, str]:
    print(f"{label} backend: eager (compile disabled)")
    return module, "eager"


def _dtype_tag(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.float32:
        return "fp32"
    raise ValueError(f"Unsupported TensorRT VAE dtype: {dtype}")


def _get_vae_scale_factor(vae: nn.Module) -> int:
    sample_tile = int(getattr(vae, "tile_sample_min_size"))
    latent_tile = int(getattr(vae, "tile_latent_min_size"))
    if sample_tile % latent_tile != 0:
        raise RuntimeError(
            f"Unsupported VAE tile ratio: sample_tile={sample_tile}, latent_tile={latent_tile}"
        )
    return sample_tile // latent_tile


def _compute_full_tile_starts(total_size: int, tile_size: int) -> List[int]:
    if total_size <= 0:
        raise ValueError(f"total_size must be positive, got {total_size}")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if total_size <= tile_size:
        return [0]

    num_tiles = math.ceil(total_size / tile_size)
    max_start = total_size - tile_size
    starts: List[int] = []

    for index in range(num_tiles):
        if index == 0:
            start = 0
        elif index == num_tiles - 1:
            start = max_start
        else:
            start = round(index * max_start / (num_tiles - 1))
            min_start = starts[-1] + 1
            remaining_slots = (num_tiles - 1) - index
            max_allowed = max_start - remaining_slots
            start = max(min_start, min(start, max_allowed))
        starts.append(start)

    return starts


def _compute_axis_overlaps(starts: List[int], tile_size: int) -> List[Tuple[int, int]]:
    overlaps: List[Tuple[int, int]] = []
    for index, start in enumerate(starts):
        left_overlap = 0
        right_overlap = 0
        if index > 0:
            left_overlap = starts[index - 1] + tile_size - start
        if index + 1 < len(starts):
            right_overlap = start + tile_size - starts[index + 1]
        if left_overlap < 0 or right_overlap < 0:
            raise RuntimeError("Tile starts must be monotonically increasing.")
        if left_overlap >= tile_size or right_overlap >= tile_size:
            raise RuntimeError(
                f"Tile overlap must be smaller than tile size, got left={left_overlap}, "
                f"right={right_overlap}, tile_size={tile_size}"
            )
        overlaps.append((left_overlap, right_overlap))
    return overlaps


def _axis_blend_weight(
    tile_size: int,
    left_overlap: int,
    right_overlap: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    weight = torch.ones(tile_size, device=device, dtype=dtype)
    if left_overlap > 0:
        weight[:left_overlap] = torch.arange(left_overlap, device=device, dtype=dtype) / left_overlap
    if right_overlap > 0:
        weight[-right_overlap:] = torch.arange(right_overlap, 0, -1, device=device, dtype=dtype) / right_overlap
    return weight


def _can_use_static_sample_tiling(vae: nn.Module, height: int, width: int) -> bool:
    scale = _get_vae_scale_factor(vae)
    return height > 0 and width > 0 and height % scale == 0 and width % scale == 0


def _can_use_static_latent_tiling(vae: nn.Module, latent_height: int, latent_width: int) -> bool:
    return latent_height > 0 and latent_width > 0


def _get_static_tiling_geometry(
    input_tile_size: int,
    output_tile_size: int,
    overlap_factor: float,
    label: str,
) -> Tuple[int, int, int]:
    overlap_size = int(input_tile_size * (1 - overlap_factor))
    blend_extent = int(output_tile_size * overlap_factor)
    row_limit = output_tile_size - blend_extent
    if overlap_size <= 0:
        raise ValueError(
            f"Static TensorRT {label} tiling requires a positive overlap step, "
            f"got input_tile_size={input_tile_size}, overlap_factor={overlap_factor}"
        )
    if row_limit <= 0:
        raise ValueError(
            f"Static TensorRT {label} tiling requires a positive row_limit, "
            f"got output_tile_size={output_tile_size}, overlap_factor={overlap_factor}"
        )
    return overlap_size, blend_extent, row_limit


def _static_overlap_tiled_encode(vae: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if not _can_use_static_sample_tiling(vae, x.shape[2], x.shape[3]):
        scale = _get_vae_scale_factor(vae)
        raise ValueError(
            f"Static TensorRT tiled encode requires input H/W to be divisible by {scale}, "
            f"got {tuple(x.shape[-2:])}"
        )

    scale = _get_vae_scale_factor(vae)
    tile_sample_size = int(getattr(vae, "tile_sample_min_size"))
    tile_latent_size = int(getattr(vae, "tile_latent_min_size"))
    latent_height = x.shape[2] // scale
    latent_width = x.shape[3] // scale
    overlap_factor = float(getattr(vae, "tile_overlap_factor", 0.25))
    overlap_size, blend_extent, row_limit = _get_static_tiling_geometry(
        input_tile_size=tile_sample_size,
        output_tile_size=tile_latent_size,
        overlap_factor=overlap_factor,
        label="encode",
    )

    rows = []
    for sample_top in range(0, x.shape[2], overlap_size):
        row = []
        for sample_left in range(0, x.shape[3], overlap_size):
            tile = x[:, :, sample_top : sample_top + tile_sample_size, sample_left : sample_left + tile_sample_size]
            tile = vae.encoder(tile)
            if vae.config.use_quant_conv:
                tile = vae.quant_conv(tile)
            row.append(tile)
        rows.append(row)

    if not rows:
        raise RuntimeError("Static tiled encode did not produce any tiles.")

    # Keep the original diffusers tile traversal and blend/crop policy so the
    # static TRT wrapper only changes how short tiles are executed, not how the
    # latent canvas is assembled.
    result_rows = []
    for row_index, row in enumerate(rows):
        result_row = []
        for col_index, tile in enumerate(row):
            if row_index > 0:
                tile = vae.blend_v(rows[row_index - 1][col_index], tile, blend_extent)
            if col_index > 0:
                tile = vae.blend_h(row[col_index - 1], tile, blend_extent)
            result_row.append(tile[:, :, :row_limit, :row_limit])
        result_rows.append(torch.cat(result_row, dim=3))

    latent = torch.cat(result_rows, dim=2)
    return latent[:, :, :latent_height, :latent_width]


def _static_overlap_tiled_decode_impl(vae: nn.Module, z: torch.Tensor) -> torch.Tensor:
    if not _can_use_static_latent_tiling(vae, z.shape[2], z.shape[3]):
        raise ValueError(f"Static TensorRT tiled decode requires positive latent H/W, got {tuple(z.shape[-2:])}")

    scale = _get_vae_scale_factor(vae)
    tile_sample_size = int(getattr(vae, "tile_sample_min_size"))
    tile_latent_size = int(getattr(vae, "tile_latent_min_size"))
    sample_height = z.shape[2] * scale
    sample_width = z.shape[3] * scale
    overlap_factor = float(getattr(vae, "tile_overlap_factor", 0.25))
    overlap_size, blend_extent, row_limit = _get_static_tiling_geometry(
        input_tile_size=tile_latent_size,
        output_tile_size=tile_sample_size,
        overlap_factor=overlap_factor,
        label="decode",
    )

    rows = []
    for latent_top in range(0, z.shape[2], overlap_size):
        row = []
        for latent_left in range(0, z.shape[3], overlap_size):
            tile = z[:, :, latent_top : latent_top + tile_latent_size, latent_left : latent_left + tile_latent_size]
            if vae.config.use_post_quant_conv:
                tile = vae.post_quant_conv(tile)
            row.append(vae.decoder(tile))
        rows.append(row)

    if not rows:
        raise RuntimeError("Static tiled decode did not produce any tiles.")

    result_rows = []
    for row_index, row in enumerate(rows):
        result_row = []
        for col_index, tile in enumerate(row):
            if row_index > 0:
                tile = vae.blend_v(rows[row_index - 1][col_index], tile, blend_extent)
            if col_index > 0:
                tile = vae.blend_h(row[col_index - 1], tile, blend_extent)
            result_row.append(tile[:, :, :row_limit, :row_limit])
        result_rows.append(torch.cat(result_row, dim=3))

    decoded = torch.cat(result_rows, dim=2)
    return decoded[:, :, :sample_height, :sample_width]


def _patched_tiled_encode(self, x: torch.Tensor, return_dict: bool = True) -> AutoencoderKLOutput:
    deprecation_message = (
        "The tiled_encode implementation supporting the `return_dict` parameter is deprecated. In the future, the "
        "implementation of this method will be replaced with that of `_tiled_encode` and you will no longer be able "
        "to pass `return_dict`. You will also have to create a `DiagonalGaussianDistribution()` from the returned value."
    )
    deprecate("tiled_encode", "1.0.0", deprecation_message, standard_warn=False)

    moments = _static_overlap_tiled_encode(self, x)
    posterior = DiagonalGaussianDistribution(moments)

    if not return_dict:
        return (posterior,)

    return AutoencoderKLOutput(latent_dist=posterior)


def _patched_tiled_decode(self, z: torch.Tensor, return_dict: bool = True):
    decoded = _static_overlap_tiled_decode_impl(self, z)
    if not return_dict:
        return (decoded,)
    return DecoderOutput(sample=decoded)


def _install_static_tiling_hooks(vae: nn.Module) -> None:
    if hasattr(vae, "_gpu_inference_static_tiling_installed"):
        return

    vae._tiled_encode = MethodType(lambda self, x: _static_overlap_tiled_encode(self, x), vae)
    vae.tiled_encode = MethodType(_patched_tiled_encode, vae)
    vae.tiled_decode = MethodType(_patched_tiled_decode, vae)
    vae._gpu_inference_static_tiling_installed = True


class TensorRTVAEEncoderWrapper(nn.Module):
    def __init__(self, trt_module: nn.Module, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.trt_module = trt_module
        self._runtime_anchor = nn.Parameter(torch.empty(0, device=device, dtype=dtype), requires_grad=False)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = sample.to(device=self._runtime_anchor.device, dtype=self._runtime_anchor.dtype)
        return self.trt_module(sample)


class TensorRTVAEDecoderWrapper(nn.Module):
    def __init__(
        self,
        trt_module: nn.Module,
        device: torch.device,
        input_dtype: torch.dtype,
        output_dtype: torch.dtype,
    ):
        super().__init__()
        self.trt_module = trt_module
        self._runtime_anchor = nn.Parameter(torch.empty(0, device=device, dtype=input_dtype), requires_grad=False)
        self.output_dtype = output_dtype

    def forward(self, sample: torch.Tensor, latent_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        if latent_embeds is not None:
            raise ValueError("TensorRT VAE decoder wrapper does not support latent_embeds.")
        sample = sample.to(device=self._runtime_anchor.device, dtype=self._runtime_anchor.dtype)
        result = self.trt_module(sample)
        if result.dtype != self.output_dtype:
            result = result.to(dtype=self.output_dtype)
        return result


class StaticVAEEncoderWrapper(nn.Module):
    def __init__(
        self,
        static_encoder: nn.Module,
        tile_sample_size: int,
        tile_latent_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.static_encoder = static_encoder
        self.tile_sample_size = tile_sample_size
        self.tile_latent_size = tile_latent_size
        self._runtime_anchor = nn.Parameter(torch.empty(0, device=device, dtype=dtype), requires_grad=False)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = sample.to(device=self._runtime_anchor.device, dtype=self._runtime_anchor.dtype)
        height, width = sample.shape[-2:]
        if height > self.tile_sample_size or width > self.tile_sample_size:
            raise ValueError(
                f"Static VAE encoder only supports tiles up to {self.tile_sample_size}, "
                f"got {tuple(sample.shape[-2:])}"
            )

        pad_h = self.tile_sample_size - height
        pad_w = self.tile_sample_size - width
        if pad_h or pad_w:
            sample = F.pad(sample, (0, pad_w, 0, pad_h))

        latent = self.static_encoder(sample)
        scale = self.tile_sample_size // self.tile_latent_size
        if height % scale != 0 or width % scale != 0:
            raise ValueError(
                f"Static VAE encoder expects H/W divisible by {scale}, got {tuple((height, width))}"
            )
        return latent[:, :, : height // scale, : width // scale]


class StaticVAEDecoderWrapper(nn.Module):
    def __init__(
        self,
        static_decoder: nn.Module,
        tile_latent_size: int,
        tile_sample_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.static_decoder = static_decoder
        self.tile_latent_size = tile_latent_size
        self.tile_sample_size = tile_sample_size
        self._runtime_anchor = nn.Parameter(torch.empty(0, device=device, dtype=dtype), requires_grad=False)

    def forward(self, sample: torch.Tensor, latent_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        sample = sample.to(device=self._runtime_anchor.device, dtype=self._runtime_anchor.dtype)
        if latent_embeds is not None:
            raise ValueError("Static TensorRT VAE decoder does not support latent_embeds.")

        latent_height, latent_width = sample.shape[-2:]
        if latent_height > self.tile_latent_size or latent_width > self.tile_latent_size:
            raise ValueError(
                f"Static VAE decoder only supports latent tiles up to {self.tile_latent_size}, "
                f"got {tuple(sample.shape[-2:])}"
            )

        pad_h = self.tile_latent_size - latent_height
        pad_w = self.tile_latent_size - latent_width
        if pad_h or pad_w:
            sample = F.pad(sample, (0, pad_w, 0, pad_h))

        decoded = self.static_decoder(sample)
        scale = self.tile_sample_size // self.tile_latent_size
        return decoded[:, :, : latent_height * scale, : latent_width * scale]


class DecoderAttentionUpcastWrapper(nn.Module):
    """Numerically stable decoder wrapper for TRT export.

    The first VAE mid-block attention overflows when TensorRT executes the whole
    decoder in pure FP16. Upcasting only that attention path to FP32 preserves
    finite outputs while keeping the rest of the decoder in FP16.
    """

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.conv_in = copy.deepcopy(decoder.conv_in).to("cuda", dtype=torch.float16)
        self.resnet0 = copy.deepcopy(decoder.mid_block.resnets[0]).to("cuda", dtype=torch.float16)
        self.attn0 = copy.deepcopy(decoder.mid_block.attentions[0]).to("cuda", dtype=torch.float32)
        self.resnet1 = copy.deepcopy(decoder.mid_block.resnets[1]).to("cuda", dtype=torch.float16)
        self.up_blocks = copy.deepcopy(decoder.up_blocks).to("cuda", dtype=torch.float16)
        self.conv_norm_out = copy.deepcopy(decoder.conv_norm_out).to("cuda", dtype=torch.float16)
        self.conv_act = copy.deepcopy(decoder.conv_act).to("cuda")
        self.conv_out = copy.deepcopy(decoder.conv_out).to("cuda", dtype=torch.float16)

    def forward(self, sample: torch.Tensor, latent_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        if latent_embeds is not None:
            raise ValueError("DecoderAttentionUpcastWrapper does not support latent_embeds.")

        sample = self.conv_in(sample)
        sample = self.resnet0(sample, None)
        sample = self.attn0(sample.float(), temb=None).to(dtype=torch.float16)
        sample = self.resnet1(sample, None)

        for up_block in self.up_blocks:
            sample = up_block(sample, None)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


def _build_static_encoder_runtime(
    encoder: nn.Module,
    vae_id: str,
    tile_sample_size: int,
    cache_dir: str,
    force_rebuild: bool,
    runtime_dtype: torch.dtype,
) -> nn.Module:
    dtype_tag = _dtype_tag(runtime_dtype)
    input_channels = int(getattr(encoder.conv_in, "in_channels"))
    dummy_inputs = (
        torch.randn(1, input_channels, tile_sample_size, tile_sample_size, device="cuda", dtype=runtime_dtype),
    )
    engine_name = f"{vae_id}_encoder_{dtype_tag}_static_{tile_sample_size}"
    return load_engine_from_pytorch(
        model=encoder,
        dummy_inputs=dummy_inputs,
        engine_name=engine_name,
        input_names=["sample"],
        output_names=["latent"],
        opset_version=19,
        precision=dtype_tag,
        force_rebuild=force_rebuild,
        cache_dir=cache_dir,
        simplify_model=False,
        dynamo_export=True,
    )


def _build_static_decoder_runtime(
    decoder: nn.Module,
    vae_id: str,
    tile_latent_size: int,
    cache_dir: str,
    force_rebuild: bool,
    runtime_dtype: torch.dtype,
    use_attention_upcast: bool = False,
) -> nn.Module:
    dtype_tag = _dtype_tag(runtime_dtype)
    latent_channels = int(getattr(decoder.conv_in, "in_channels"))
    export_model = decoder
    if use_attention_upcast:
        export_model = DecoderAttentionUpcastWrapper(decoder).eval()

    dummy_inputs = (
        torch.randn(1, latent_channels, tile_latent_size, tile_latent_size, device="cuda", dtype=runtime_dtype),
    )
    engine_name = (
        f"{vae_id}_decoder_{dtype_tag}"
        f"{'_attnupcast' if use_attention_upcast else ''}_static_{tile_latent_size}"
    )
    return load_engine_from_pytorch(
        model=export_model,
        dummy_inputs=dummy_inputs,
        engine_name=engine_name,
        input_names=["latent"],
        output_names=["sample"],
        opset_version=19,
        precision=dtype_tag,
        force_rebuild=force_rebuild,
        cache_dir=cache_dir,
        simplify_model=False,
        dynamo_export=True,
    )


def _validate_trt_module(
    module: nn.Module,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    label: str,
) -> None:
    with torch.inference_mode():
        probe = module(torch.randn(*shape, device=device, dtype=dtype))
    if not torch.isfinite(probe).all():
        raise RuntimeError(f"{label} self-check failed: non-finite values detected.")


def _enable_trt_runtime(
    vae: nn.Module,
    vae_id: str,
    cache_dir: str,
    force_rebuild: bool,
) -> Dict[str, str]:
    device, dtype = _get_module_device_and_dtype(vae.encoder)
    if device.type != "cuda":
        raise RuntimeError("TensorRT VAE backend requires CUDA.")
    if dtype != torch.float16:
        raise RuntimeError(f"TensorRT VAE backend requires float16 modules, got {dtype}.")

    tile_sample_size = int(getattr(vae, "tile_sample_min_size"))
    tile_latent_size = int(getattr(vae, "tile_latent_min_size"))

    encoder = vae.encoder.eval().to(device=device, dtype=torch.float16)
    decoder = vae.decoder.eval().to(device=device, dtype=torch.float16)
    encoder_input_channels = int(getattr(encoder.conv_in, "in_channels"))
    decoder_input_channels = int(getattr(decoder.conv_in, "in_channels"))

    static_trt_encoder = _build_static_encoder_runtime(
        encoder=encoder,
        vae_id=vae_id,
        tile_sample_size=tile_sample_size,
        cache_dir=cache_dir,
        force_rebuild=force_rebuild,
        runtime_dtype=torch.float16,
    )
    if not _env_flag(VAE_TRT_SKIP_SELF_CHECK_ENV, False):
        _validate_trt_module(
            static_trt_encoder,
            shape=(1, encoder_input_channels, tile_sample_size, tile_sample_size),
            dtype=torch.float16,
            device=device,
            label="TensorRT VAE encoder static",
        )
    else:
        print(f"[VAE] Skipping TensorRT self-check because {VAE_TRT_SKIP_SELF_CHECK_ENV}=1")

    decoder_backend = "trt-static-fp16"
    dtype_tag = _dtype_tag(torch.float16)
    attnupcast_plan_path = os.path.join(
        cache_dir,
        f"{vae_id}_decoder_{dtype_tag}_attnupcast_static_{tile_latent_size}_{dtype_tag}.plan",
    )
    if os.path.exists(attnupcast_plan_path) and not force_rebuild:
        static_trt_decoder = _build_static_decoder_runtime(
            decoder=decoder,
            vae_id=vae_id,
            tile_latent_size=tile_latent_size,
            cache_dir=cache_dir,
            force_rebuild=force_rebuild,
            runtime_dtype=torch.float16,
            use_attention_upcast=True,
        )
        if not _env_flag(VAE_TRT_SKIP_SELF_CHECK_ENV, False):
            _validate_trt_module(
                static_trt_decoder,
                shape=(1, decoder_input_channels, tile_latent_size, tile_latent_size),
                dtype=torch.float16,
                device=device,
                label="TensorRT VAE decoder static fp16-attn-upcast",
            )
        decoder_backend = "trt-static-fp16-attn-upcast"
    else:
        try:
            static_trt_decoder = _build_static_decoder_runtime(
                decoder=decoder,
                vae_id=vae_id,
                tile_latent_size=tile_latent_size,
                cache_dir=cache_dir,
                force_rebuild=force_rebuild,
                runtime_dtype=torch.float16,
                use_attention_upcast=False,
            )
            if not _env_flag(VAE_TRT_SKIP_SELF_CHECK_ENV, False):
                _validate_trt_module(
                    static_trt_decoder,
                    shape=(1, decoder_input_channels, tile_latent_size, tile_latent_size),
                    dtype=torch.float16,
                    device=device,
                    label="TensorRT VAE decoder static fp16",
                )
        except Exception as exc:
            print(f"[VAE] Decoder static FP16 TensorRT failed under StrongTyped: {exc}")
            static_trt_decoder = _build_static_decoder_runtime(
                decoder=decoder,
                vae_id=vae_id,
                tile_latent_size=tile_latent_size,
                cache_dir=cache_dir,
                force_rebuild=force_rebuild,
                runtime_dtype=torch.float16,
                use_attention_upcast=True,
            )
            if not _env_flag(VAE_TRT_SKIP_SELF_CHECK_ENV, False):
                _validate_trt_module(
                    static_trt_decoder,
                    shape=(1, decoder_input_channels, tile_latent_size, tile_latent_size),
                    dtype=torch.float16,
                    device=device,
                    label="TensorRT VAE decoder static fp16-attn-upcast",
                )
            decoder_backend = "trt-static-fp16-attn-upcast"
        else:
            decoder_backend = "trt-static-fp16"

    if decoder_backend == "trt-static-fp16-attn-upcast" and _env_flag(VAE_TRT_SKIP_SELF_CHECK_ENV, False):
        print(f"[VAE] Skipping TensorRT decoder self-check because {VAE_TRT_SKIP_SELF_CHECK_ENV}=1")
    elif decoder_backend == "trt-static-fp16" and _env_flag(VAE_TRT_SKIP_SELF_CHECK_ENV, False):
        print(f"[VAE] Skipping TensorRT decoder self-check because {VAE_TRT_SKIP_SELF_CHECK_ENV}=1")

    vae.encoder = StaticVAEEncoderWrapper(
        static_encoder=TensorRTVAEEncoderWrapper(static_trt_encoder, device=device, dtype=torch.float16).eval(),
        tile_sample_size=tile_sample_size,
        tile_latent_size=tile_latent_size,
        device=device,
        dtype=torch.float16,
    ).eval()
    vae.decoder = StaticVAEDecoderWrapper(
        static_decoder=TensorRTVAEDecoderWrapper(
            static_trt_decoder,
            device=device,
            input_dtype=torch.float16,
            output_dtype=torch.float16,
        ).eval(),
        tile_latent_size=tile_latent_size,
        tile_sample_size=tile_sample_size,
        device=device,
        dtype=torch.float16,
    ).eval()
    del encoder, decoder, static_trt_encoder, static_trt_decoder
    _install_static_tiling_hooks(vae)
    _release_torch_cuda_cache()

    return {
        "backend": "trt-static-tiling",
        "encoder_backend": "trt-static",
        "decoder_backend": decoder_backend,
        "short_tile_policy": "pad-crop-static",
        "tiling_backend": "diffusers-legacy-blend",
        "cache_dir": cache_dir,
    }


def configure_fp16_vae_runtime(
    vae: nn.Module,
    *,
    vae_id: str = "ae",
    cache_dir: Optional[str] = None,
    use_trt: Optional[bool] = None,
    force_rebuild: Optional[bool] = None,
    compile_mode: str = "default",
) -> Tuple[nn.Module, Dict[str, str]]:
    cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "vae", "engines")
    os.makedirs(cache_dir, exist_ok=True)

    use_trt = _env_flag(VAE_TRT_ENV, True) if use_trt is None else use_trt
    force_rebuild = _env_flag(VAE_TRT_FORCE_REBUILD_ENV, False) if force_rebuild is None else force_rebuild

    if use_trt:
        try:
            backend_info = _enable_trt_runtime(
                vae=vae,
                vae_id=vae_id,
                cache_dir=cache_dir,
                force_rebuild=force_rebuild,
            )
            print(f"[VAE] Runtime backend: TensorRT static tiling (cache: {cache_dir})")
            vae._gpu_inference_runtime_backend = backend_info
            return vae, backend_info
        except Exception as exc:
            raise RuntimeError("TensorRT VAE setup failed; explicit eager routing is required") from exc

    encoder_backend = "eager"
    decoder_backend = "eager"
    vae.encoder, encoder_backend = _maybe_compile_module(vae.encoder, "[VAE Encoder]", mode=compile_mode)
    vae.decoder, decoder_backend = _maybe_compile_module(vae.decoder, "[VAE Decoder]", mode=compile_mode)

    backend = "torch-compile" if encoder_backend == "compile" and decoder_backend == "compile" else "torch"
    backend_info = {
        "backend": backend,
        "encoder_backend": encoder_backend,
        "decoder_backend": decoder_backend,
        "cache_dir": cache_dir,
    }
    print(
        f"[VAE] Runtime backend: {backend} "
        f"(encoder={encoder_backend}, decoder={decoder_backend})"
    )
    vae._gpu_inference_runtime_backend = backend_info
    return vae, backend_info
