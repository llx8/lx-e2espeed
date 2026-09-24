"""CUDA JPEG decode/encode bridge extracted from production service code."""

from __future__ import annotations

import numpy as np
import torch
from nvidia import nvimgcodec
from torch.utils.dlpack import to_dlpack


class NvJpegTorchCodec:
    """Keep nvImageCodec objects resident and exchange image data with PyTorch."""

    def __init__(
        self,
        *,
        quality: int = 98,
        chroma_subsampling=nvimgcodec.CSS_420,
        fancy_upsampling: bool = True,
    ) -> None:
        options = ":fancy_upsampling=1" if fancy_upsampling else ":fancy_upsampling=0"
        self.decoder = nvimgcodec.Decoder(options=options)
        self.decode_params = nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.SRGB)
        self.encoder = nvimgcodec.Encoder()
        self.encode_params = nvimgcodec.EncodeParams(
            quality_type=nvimgcodec.QualityType.QUALITY,
            quality_value=int(quality),
            color_spec=nvimgcodec.ColorSpec.SRGB,
            chroma_subsampling=chroma_subsampling,
        )

    @staticmethod
    def _stream(device: torch.device | None = None) -> int:
        if not torch.cuda.is_available():
            raise RuntimeError("nvImageCodec JPEG path requires CUDA")
        return torch.cuda.current_stream(device=device).cuda_stream

    def decode_to_bgr(self, jpeg_bytes: bytes) -> torch.Tensor:
        """Decode JPEG bytes to a contiguous CUDA HWC uint8 BGR tensor."""
        image_buffer = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        cuda_stream = self._stream()
        decoded = self.decoder.decode(
            image_buffer,
            params=self.decode_params,
            cuda_stream=cuda_stream,
        )
        image = torch.from_dlpack(decoded)
        if not image.is_cuda:
            raise RuntimeError("nvImageCodec decoder returned a non-CUDA image")
        if image.ndim == 2:
            image = image.unsqueeze(-1)

        channels = int(image.shape[-1])
        if channels == 1:
            image = image.repeat(1, 1, 3)
        elif channels >= 3:
            image = image[..., :3][..., [2, 1, 0]]
        else:
            raise ValueError(f"Unsupported decoded channel count: {channels}")

        image = image.contiguous()
        image._nvimgcodec_owner = decoded
        return image

    def encode_bgr(self, image_bgr: torch.Tensor) -> bytes:
        """Encode a CUDA HWC BGR tensor to JPEG bytes."""
        if not isinstance(image_bgr, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(image_bgr).__name__}")
        if not image_bgr.is_cuda:
            raise ValueError("nvImageCodec encoder input must already be on CUDA")
        if image_bgr.ndim == 2:
            image_rgb = image_bgr.unsqueeze(-1).repeat(1, 1, 3)
        elif image_bgr.ndim == 3 and int(image_bgr.shape[-1]) >= 3:
            image_rgb = image_bgr[..., :3][..., [2, 1, 0]]
        else:
            raise ValueError(f"Expected HWC BGR image, got shape {tuple(image_bgr.shape)}")

        if torch.is_floating_point(image_rgb):
            image_rgb = image_rgb.mul(255).clamp(0, 255).to(torch.uint8)
        elif image_rgb.dtype != torch.uint8:
            image_rgb = image_rgb.clamp(0, 255).to(torch.uint8)
        image_rgb = image_rgb.contiguous()

        cuda_stream = self._stream(image_rgb.device)
        nv_image = nvimgcodec.as_image(
            to_dlpack(image_rgb),
            sample_format=nvimgcodec.SampleFormat.I_RGB,
            color_spec=nvimgcodec.ColorSpec.SRGB,
            cuda_stream=cuda_stream,
        )
        code_stream = self.encoder.encode(
            nv_image,
            codec="jpeg",
            params=self.encode_params,
            cuda_stream=cuda_stream,
        )
        torch.cuda.synchronize(device=image_rgb.device)
        return np.asarray(code_stream).tobytes()
