#!/usr/bin/env python3
"""Fill LoRA converter.

Accepts two input formats:
1. Raw Kontext/Nunchaku format (`double_blocks.*.processor.*.down/up.weight`),
   converted directly into a diffusers LoRA that V2 `update_lora_params_v2` can load.
2. Older diffusers fused-proj_out format (`*.lora_A/B.weight`),
   keeping the original SVD repair logic; can export at a target rank.
"""

import argparse
import os
import re
from collections import defaultdict
from typing import Optional

import safetensors.torch
import torch
from safetensors import safe_open
from tqdm import tqdm

ATTN_DIM = 3072
MLP_DIM = 12288
FUSED_DIM = ATTN_DIM + MLP_DIM


def detect_lora_format(input_path: str) -> str:
    with safe_open(input_path, framework="pt") as handle:
        keys = list(handle.keys())

    if any(".processor." in key and key.endswith((".down.weight", ".up.weight")) for key in keys):
        return "kontext_raw"
    if any(".lora_A.weight" in key or ".lora_B.weight" in key for key in keys):
        return "diffusers"
    raise ValueError(f"unrecognized LoRA format: {input_path}")


def svd_decompose(delta: torch.Tensor, target_rank: int):
    matrix = delta.float()
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)

    valid_rank = min(target_rank, s.shape[0])
    u = u[:, :valid_rank]
    s = s[:valid_rank]
    vh = vh[:valid_rank, :]

    sqrt_s = torch.diag(torch.sqrt(s))
    lora_b = u @ sqrt_s
    lora_a = sqrt_s @ vh
    return lora_a, lora_b


def convert_diffusers_lora(input_path: str, target_rank: int, output_path: str):
    print(f"\n{'=' * 60}")
    print(f"  converting diffusers Fill LoRA -> rank={target_rank}")
    print(f"  input: {input_path}")
    print(f"  output: {output_path}")
    print(f"{'=' * 60}")

    sd = safetensors.torch.load_file(input_path)

    pairs = defaultdict(dict)
    for key, tensor in sd.items():
        if ".lora_A.weight" in key:
            base = key.replace(".lora_A.weight", "")
            pairs[base]["A"] = tensor
        elif ".lora_B.weight" in key:
            base = key.replace(".lora_B.weight", "")
            pairs[base]["B"] = tensor

    new_sd = {}
    proj_out_count = 0
    regular_count = 0
    skipped_count = 0

    for base_key, ab in tqdm(sorted(pairs.items()), desc="Processing"):
        a_tensor = ab.get("A")
        b_tensor = ab.get("B")
        if a_tensor is None or b_tensor is None:
            skipped_count += 1
            continue

        is_proj_out = bool(re.search(r"single_transformer_blocks\.\d+\.proj_out$", base_key))

        if is_proj_out:
            delta_full = b_tensor.float() @ a_tensor.float()
            if delta_full.shape != (ATTN_DIM, FUSED_DIM):
                raise ValueError(f"unexpected proj_out delta shape: {base_key} {delta_full.shape}")

            delta_attn = delta_full[:, :ATTN_DIM]
            delta_mlp = delta_full[:, ATTN_DIM:]

            a_attn, b_attn = svd_decompose(delta_attn, target_rank)
            a_mlp, b_mlp = svd_decompose(delta_mlp, target_rank)

            r_attn = a_attn.shape[0]
            r_mlp = a_mlp.shape[0]
            total_rank = r_attn + r_mlp

            a_fused = torch.zeros(total_rank, FUSED_DIM, dtype=torch.bfloat16)
            a_fused[:r_attn, :ATTN_DIM] = a_attn.to(torch.bfloat16)
            a_fused[r_attn:, ATTN_DIM:] = a_mlp.to(torch.bfloat16)

            b_fused = torch.zeros(ATTN_DIM, total_rank, dtype=torch.bfloat16)
            b_fused[:, :r_attn] = b_attn.to(torch.bfloat16)
            b_fused[:, r_attn:] = b_mlp.to(torch.bfloat16)

            new_sd[f"{base_key}.lora_A.weight"] = a_fused
            new_sd[f"{base_key}.lora_B.weight"] = b_fused
            proj_out_count += 1
        else:
            delta = b_tensor.float() @ a_tensor.float()
            if delta.abs().max() < 1e-8:
                skipped_count += 1
                continue

            new_a, new_b = svd_decompose(delta, target_rank)
            new_sd[f"{base_key}.lora_A.weight"] = new_a.to(torch.bfloat16)
            new_sd[f"{base_key}.lora_B.weight"] = new_b.to(torch.bfloat16)
            regular_count += 1

    print("\n  statistics:")
    print(f"    proj_out (split and regrouped): {proj_out_count}")
    print(f"    regular keys:       {regular_count}")
    print(f"    skipped:            {skipped_count}")
    print(f"    total keys:         {len(new_sd)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    safetensors.torch.save_file(new_sd, output_path)
    print(f"\n  saved: {output_path}")
    return new_sd


def convert_raw_kontext_lora(input_path: str, output_path: str):
    print(f"\n{'=' * 60}")
    print("  converting raw Kontext Fill LoRA -> V2 diffusers LoRA")
    print(f"  input: {input_path}")
    print(f"  output: {output_path}")
    print(f"{'=' * 60}")

    from nunchaku.lora.flux import nunchaku_converter

    converted = nunchaku_converter.to_diffusers(input_path)
    converted = {key: value.clone().contiguous() for key, value in converted.items()}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    safetensors.torch.save_file(converted, output_path)
    print(f"\n  saved: {output_path}")
    return converted


def validate_v2_compatibility(output_path: str):
    from nunchaku.lora.flux.v1.lora_flux_v2 import _classify_and_map_key

    with safe_open(output_path, framework="pt") as handle:
        keys = list(handle.keys())

    parsed = sum(1 for key in keys if _classify_and_map_key(key) is not None)
    print("\n  V2 compatibility check:")
    print(f"    total keys:  {len(keys)}")
    print(f"    parsed keys:  {parsed}")
    print(f"    unparsed keys:  {len(keys) - parsed}")

    if parsed != len(keys):
        raise ValueError(f"V2 compatibility check failed: {output_path}")


def convert_fill_lora(
    input_path: str,
    output_path: str,
    target_rank: Optional[int] = None,
    mode: str = "auto",
):
    detected_mode = detect_lora_format(input_path) if mode == "auto" else mode

    if detected_mode == "kontext_raw":
        converted = convert_raw_kontext_lora(input_path, output_path)
    elif detected_mode == "diffusers":
        if target_rank is None:
            raise ValueError("the diffusers format needs --target-rank")
        converted = convert_diffusers_lora(input_path, target_rank, output_path)
    else:
        raise ValueError(f"unsupported mode: {detected_mode}")

    validate_v2_compatibility(output_path)
    return converted


def main():
    parser = argparse.ArgumentParser(description="convert or compress a Fill LoRA so V2 update_lora_params_v2 can load it")
    parser.add_argument("--input", required=True, help="input LoRA path")
    parser.add_argument("--output", required=True, help="output LoRA path")
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "kontext_raw", "diffusers"],
        help="input weight format; detected automatically by default",
    )
    parser.add_argument("--target-rank", type=int, default=None, help="needed only in diffusers mode")
    args = parser.parse_args()

    with torch.no_grad():
        convert_fill_lora(
            input_path=args.input,
            output_path=args.output,
            target_rank=args.target_rank,
            mode=args.mode,
        )

    print("\ndone.")


if __name__ == "__main__":
    main()
