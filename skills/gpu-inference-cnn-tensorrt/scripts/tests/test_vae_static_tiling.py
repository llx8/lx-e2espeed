import torch
import torch.nn as nn
from types import SimpleNamespace
from pathlib import Path
import sys

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR / "runtime"))
sys.path.insert(0, str(SCRIPTS_DIR / "examples"))

from vae_tensorrt import (
    StaticVAEDecoderWrapper,
    StaticVAEEncoderWrapper,
    _axis_blend_weight,
    _compute_axis_overlaps,
    _compute_full_tile_starts,
    _static_overlap_tiled_decode_impl,
    _static_overlap_tiled_encode,
)


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-6) -> None:
    if not torch.allclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"{name} mismatch:\nactual={actual}\nexpected={expected}")


def test_full_tile_starts() -> None:
    cases = {
        (128, 128): [0],
        (129, 128): [0, 1],
        (130, 128): [0, 2],
        (192, 128): [0, 64],
        (224, 128): [0, 96],
        (256, 128): [0, 128],
        (257, 128): [0, 64, 129],
    }
    for (total, tile), expected in cases.items():
        actual = _compute_full_tile_starts(total, tile)
        if actual != expected:
            raise AssertionError(f"starts mismatch for total={total}, tile={tile}: {actual} != {expected}")


def test_axis_weight_partition() -> None:
    total = 257
    tile = 128
    starts = _compute_full_tile_starts(total, tile)
    overlaps = _compute_axis_overlaps(starts, tile)
    canvas = torch.zeros(total, dtype=torch.float32)

    for start, (left_overlap, right_overlap) in zip(starts, overlaps):
        weight = _axis_blend_weight(tile, left_overlap, right_overlap, device=torch.device("cpu"), dtype=torch.float32)
        canvas[start : start + tile] += weight

    assert_close("axis partition", canvas, torch.ones_like(canvas))


class DummyStaticEncoder(nn.Module):
    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = sample.shape
        latent = torch.arange(height * width // 16, dtype=sample.dtype, device=sample.device)
        latent = latent.view(1, 1, height // 4, width // 4)
        return latent.repeat(batch, 2, 1, 1)


class DummyStaticDecoder(nn.Module):
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = latent.shape
        sample = torch.arange(height * width * 16, dtype=latent.dtype, device=latent.device)
        sample = sample.view(1, 1, height * 4, width * 4)
        return sample.repeat(batch, 3, 1, 1)


def test_static_wrappers_pad_crop() -> None:
    encoder = StaticVAEEncoderWrapper(
        static_encoder=DummyStaticEncoder(),
        tile_sample_size=8,
        tile_latent_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    sample = torch.ones((1, 3, 8, 4), dtype=torch.float32)
    latent = encoder(sample)
    if latent.shape != (1, 2, 2, 1):
        raise AssertionError(f"unexpected latent shape: {latent.shape}")

    decoder = StaticVAEDecoderWrapper(
        static_decoder=DummyStaticDecoder(),
        tile_latent_size=2,
        tile_sample_size=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    recon = decoder(latent)
    if recon.shape != (1, 3, 8, 4):
        raise AssertionError(f"unexpected decode shape: {recon.shape}")


class DummyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.tile_sample_min_size = 8
        self.tile_latent_min_size = 2
        self.tile_overlap_factor = 0.0
        self.config = SimpleNamespace(use_quant_conv=False, use_post_quant_conv=False)
        self.encoder = StaticVAEEncoderWrapper(
            static_encoder=DummyStaticEncoder(),
            tile_sample_size=8,
            tile_latent_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.decoder = StaticVAEDecoderWrapper(
            static_decoder=DummyStaticDecoder(),
            tile_latent_size=2,
            tile_sample_size=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (
                x / blend_extent
            )
        return b


class CoordinateSensitiveEncoder(nn.Module):
    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        height, width = sample.shape[-2:]
        y = torch.arange(height, dtype=sample.dtype, device=sample.device).view(1, 1, height, 1)
        x = torch.arange(width, dtype=sample.dtype, device=sample.device).view(1, 1, 1, width)
        return sample[:, :1] + y * 10 + x


class CoordinateSensitiveDecoder(nn.Module):
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        height, width = latent.shape[-2:]
        y = torch.arange(height, dtype=latent.dtype, device=latent.device).view(1, 1, height, 1)
        x = torch.arange(width, dtype=latent.dtype, device=latent.device).view(1, 1, 1, width)
        return (latent + y * 10 + x).repeat(1, 3, 1, 1)


class DiffusersSemanticDummyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.tile_sample_min_size = 4
        self.tile_latent_min_size = 4
        self.tile_overlap_factor = 0.25
        self.config = SimpleNamespace(use_quant_conv=False, use_post_quant_conv=False)
        self.encoder = StaticVAEEncoderWrapper(
            static_encoder=CoordinateSensitiveEncoder(),
            tile_sample_size=4,
            tile_latent_size=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.decoder = StaticVAEDecoderWrapper(
            static_decoder=CoordinateSensitiveDecoder(),
            tile_latent_size=4,
            tile_sample_size=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (
                x / blend_extent
            )
        return b


def _reference_tiled_encode(vae: nn.Module, x: torch.Tensor) -> torch.Tensor:
    overlap_size = int(vae.tile_sample_min_size * (1 - vae.tile_overlap_factor))
    blend_extent = int(vae.tile_latent_min_size * vae.tile_overlap_factor)
    row_limit = vae.tile_latent_min_size - blend_extent

    rows = []
    for top in range(0, x.shape[2], overlap_size):
        row = []
        for left in range(0, x.shape[3], overlap_size):
            tile = x[:, :, top : top + vae.tile_sample_min_size, left : left + vae.tile_sample_min_size]
            tile = vae.encoder(tile)
            if vae.config.use_quant_conv:
                tile = vae.quant_conv(tile)
            row.append(tile)
        rows.append(row)

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
    return torch.cat(result_rows, dim=2)


def _reference_tiled_decode(vae: nn.Module, z: torch.Tensor) -> torch.Tensor:
    overlap_size = int(vae.tile_latent_min_size * (1 - vae.tile_overlap_factor))
    blend_extent = int(vae.tile_sample_min_size * vae.tile_overlap_factor)
    row_limit = vae.tile_sample_min_size - blend_extent

    rows = []
    for top in range(0, z.shape[2], overlap_size):
        row = []
        for left in range(0, z.shape[3], overlap_size):
            tile = z[:, :, top : top + vae.tile_latent_min_size, left : left + vae.tile_latent_min_size]
            if vae.config.use_post_quant_conv:
                tile = vae.post_quant_conv(tile)
            row.append(vae.decoder(tile))
        rows.append(row)

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
    return torch.cat(result_rows, dim=2)


def test_static_tiled_paths_shapes() -> None:
    vae = DummyVAE()
    sample = torch.ones((1, 3, 8, 12), dtype=torch.float32)
    latent = _static_overlap_tiled_encode(vae, sample)
    if latent.shape != (1, 2, 2, 3):
        raise AssertionError(f"unexpected tiled latent shape: {latent.shape}")

    recon = _static_overlap_tiled_decode_impl(vae, latent)
    if recon.shape != (1, 3, 8, 12):
        raise AssertionError(f"unexpected tiled decode shape: {recon.shape}")


def test_static_tiled_encode_matches_legacy_blend_semantics() -> None:
    vae = DiffusersSemanticDummyVAE()
    sample = torch.arange(30, dtype=torch.float32).view(1, 1, 5, 6).repeat(1, 3, 1, 1)
    actual = _static_overlap_tiled_encode(vae, sample)
    expected = _reference_tiled_encode(vae, sample)
    assert_close("static tiled encode legacy blend semantics", actual, expected)


def test_static_tiled_decode_matches_legacy_blend_semantics() -> None:
    vae = DiffusersSemanticDummyVAE()
    latent = torch.arange(30, dtype=torch.float32).view(1, 1, 5, 6)
    actual = _static_overlap_tiled_decode_impl(vae, latent)
    expected = _reference_tiled_decode(vae, latent)
    assert_close("static tiled decode legacy blend semantics", actual, expected)


def main() -> None:
    test_full_tile_starts()
    test_axis_weight_partition()
    test_static_wrappers_pad_crop()
    test_static_tiled_paths_shapes()
    test_static_tiled_encode_matches_legacy_blend_semantics()
    test_static_tiled_decode_matches_legacy_blend_semantics()
    print("test_vae_static_tiling: OK")


if __name__ == "__main__":
    main()
