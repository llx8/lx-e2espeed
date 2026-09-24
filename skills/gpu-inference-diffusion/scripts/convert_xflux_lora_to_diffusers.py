#!/usr/bin/env python3
"""Convert supported xFlux/XLabs/ai-toolkit Flux LoRA keys to diffusers."""

import argparse
import re
from typing import Dict

import torch
from safetensors.torch import load_file, save_file


def convert_xflux_lora_to_diffusers(old_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    src = dict(old_state_dict)
    new_state_dict: Dict[str, torch.Tensor] = {}
    orig_keys = list(src.keys())

    def handle_qkv(sds_sd, ait_sd, sds_key, ait_keys, dims=None):
        down_weight = sds_sd.pop(sds_key)
        up_weight = sds_sd.pop(sds_key.replace(".down.weight", ".up.weight"))

        num_splits = len(ait_keys)
        if dims is None:
            dims = [up_weight.shape[0] // num_splits] * num_splits
        else:
            assert sum(dims) == up_weight.shape[0]

        ait_down_keys = [key + ".lora_A.weight" for key in ait_keys]
        ait_up_keys = [key + ".lora_B.weight" for key in ait_keys]

        ait_sd.update(dict.fromkeys(ait_down_keys, down_weight))
        ait_sd.update({key: value for key, value in zip(ait_up_keys, torch.split(up_weight, dims, dim=0))})

    for old_key in orig_keys:
        if old_key.startswith(("diffusion_model.double_blocks", "double_blocks")):
            block_num = re.search(r"double_blocks\.(\d+)", old_key).group(1)
            new_key = f"transformer.transformer_blocks.{block_num}"

            if "processor.proj_lora1" in old_key:
                new_key += ".attn.to_out.0"
            elif "processor.proj_lora2" in old_key:
                new_key += ".attn.to_add_out"
            elif "processor.qkv_lora2" in old_key and "up" not in old_key:
                handle_qkv(
                    src,
                    new_state_dict,
                    old_key,
                    [
                        f"transformer.transformer_blocks.{block_num}.attn.add_q_proj",
                        f"transformer.transformer_blocks.{block_num}.attn.add_k_proj",
                        f"transformer.transformer_blocks.{block_num}.attn.add_v_proj",
                    ],
                )
            elif "processor.qkv_lora1" in old_key and "up" not in old_key:
                handle_qkv(
                    src,
                    new_state_dict,
                    old_key,
                    [
                        f"transformer.transformer_blocks.{block_num}.attn.to_q",
                        f"transformer.transformer_blocks.{block_num}.attn.to_k",
                        f"transformer.transformer_blocks.{block_num}.attn.to_v",
                    ],
                )

            if "down" in old_key:
                new_key += ".lora_A.weight"
            elif "up" in old_key:
                new_key += ".lora_B.weight"

        elif old_key.startswith(("diffusion_model.single_blocks", "single_blocks")):
            block_num = re.search(r"single_blocks\.(\d+)", old_key).group(1)
            new_key = f"transformer.single_transformer_blocks.{block_num}"

            if "proj_lora" in old_key:
                new_key += ".proj_out"
            elif "qkv_lora" in old_key and "up" not in old_key:
                handle_qkv(
                    src,
                    new_state_dict,
                    old_key,
                    [
                        f"transformer.single_transformer_blocks.{block_num}.attn.to_q",
                        f"transformer.single_transformer_blocks.{block_num}.attn.to_k",
                        f"transformer.single_transformer_blocks.{block_num}.attn.to_v",
                    ],
                )

            if "down" in old_key:
                new_key += ".lora_A.weight"
            elif "up" in old_key:
                new_key += ".lora_B.weight"

        else:
            new_key = old_key

        if "qkv" not in old_key:
            new_state_dict[new_key] = src.pop(old_key)

    if len(src) > 0:
        raise ValueError(f"keys left unconsumed: {list(src.keys())}")

    return {key: tensor.clone().contiguous() for key, tensor in new_state_dict.items()}


def main():
    parser = argparse.ArgumentParser(description="Convert supported xFlux/XLabs/ai-toolkit Flux LoRA to diffusers")
    parser.add_argument("--input", required=True, help="input safetensors path")
    parser.add_argument("--output", required=True, help="output safetensors path")
    args = parser.parse_args()

    loaded_tensors = load_file(args.input)
    new_state_dict = convert_xflux_lora_to_diffusers(loaded_tensors)
    save_file(new_state_dict, args.output)

    print(f"converted: {args.input}")
    print(f"output path: {args.output}")
    print(f"exported keys: {len(new_state_dict)}")


if __name__ == "__main__":
    main()
