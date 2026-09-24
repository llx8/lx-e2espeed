#!/usr/bin/env python3
"""Inspect attention seq_len, LoRA format, and rank for Flux / Kontext pipelines."""

import argparse
from collections import Counter
from typing import Dict, Iterable, List, Tuple

from safetensors import safe_open


def flux_image_seq_len(height: int, width: int, vae_scale_factor: int = 8, patch_size: int = 2) -> int:
    latent_height = height // vae_scale_factor
    latent_width = width // vae_scale_factor
    return (latent_height // patch_size) * (latent_width // patch_size)


def classify_lora_format(keys: Iterable[str]) -> str:
    key_list = list(keys)
    if any('.processor.' in key and key.endswith(('.down.weight', '.up.weight')) for key in key_list):
        return 'kontext_raw'
    if any('.lora_A.weight' in key or '.lora_B.weight' in key for key in key_list):
        return 'diffusers'
    if any(
        key.startswith(('diffusion_model.double_blocks', 'double_blocks', 'diffusion_model.single_blocks', 'single_blocks'))
        and 'lora' in key
        for key in key_list
    ):
        return 'xlabs_raw'
    return 'unknown'


def collect_lora_ranks(path: str) -> Tuple[str, List[int], int]:
    shapes: Dict[str, Tuple[int, ...]] = {}
    with safe_open(path, framework='pt') as handle:
        keys = list(handle.keys())
        for key in keys:
            shapes[key] = tuple(handle.get_tensor(key).shape)

    fmt = classify_lora_format(shapes.keys())
    ranks: List[int] = []
    for key, shape in shapes.items():
        if key.endswith('.lora_A.weight') or key.endswith('.down.weight'):
            ranks.append(shape[0])
        elif key.endswith('.lora_B.weight') or key.endswith('.up.weight'):
            if len(shape) >= 2:
                ranks.append(shape[1])

    return fmt, sorted(ranks), len(shapes)


def describe_sage_priority(total_seq_len: int) -> str:
    if total_seq_len < 2048:
        return 'Rule of thumb: the sequence is short, so SageAttention usually gains little; check dataflow and hot-path overhead first.'
    if total_seq_len < 4096:
        return 'Rule of thumb: the sequence is medium-long; SageAttention is worth a try, measured against the default attention.'
    return 'Rule of thumb: the sequence is long; SageAttention or other attention work is a high-priority candidate.'


def main():
    parser = argparse.ArgumentParser(description='inspect seq_len and LoRA rank/format for Flux / Kontext pipelines')
    parser.add_argument('--height', type=int, required=True, help='input height')
    parser.add_argument('--width', type=int, required=True, help='input width')
    parser.add_argument('--text-seq-len', type=int, default=256, help='text token length, commonly 256 for Kontext V2')
    parser.add_argument('--reference-images', type=int, default=1, help='number of reference images, commonly 1 for Kontext image editing')
    parser.add_argument('--vae-scale-factor', type=int, default=8, help='VAE downsampling factor, commonly 8 for Flux/Kontext')
    parser.add_argument('--patch-size', type=int, default=2, help='latent patch size, commonly 2 for Flux/Kontext')
    parser.add_argument('--lora', action='append', default=[], help='LoRA safetensors path to inspect; may repeat')
    args = parser.parse_args()

    image_seq = flux_image_seq_len(args.height, args.width, args.vae_scale_factor, args.patch_size)
    hidden_seq = image_seq * (1 + args.reference_images)
    total_seq = hidden_seq + args.text_seq_len

    print('=' * 72)
    print('Flux / Kontext attention length estimate')
    print('=' * 72)
    print(f'input size:              {args.height} x {args.width}')
    print(f'image_seq_len:          {image_seq}')
    print(f'image streams:          {1 + args.reference_images} (current image + reference images)')
    print(f'hidden seq len:         {hidden_seq}')
    print(f'text seq len:           {args.text_seq_len}')
    print(f'joint attention seq:    {total_seq}')
    print(f'attention cost proxy:   ~ seq^2 = {total_seq ** 2:,}')
    print(describe_sage_priority(total_seq))
    print('Note: this is an engineering estimate; whether SageAttention pays off must be measured at the target resolution.')

    for path in args.lora:
        fmt, ranks, key_count = collect_lora_ranks(path)
        print('\n' + '=' * 72)
        print(f'LoRA: {path}')
        print('=' * 72)
        print(f'format:                  {fmt}')
        print(f'total keys:              {key_count}')
        if not ranks:
            print('No rank information found; inspect the key names by hand.')
            continue

        rank_counter = Counter(ranks)
        print(f'unique ranks:           {sorted(rank_counter)}')
        print(f'max rank:               {max(ranks)}')
        print(f'rank distribution:      {dict(rank_counter)}')

        if fmt == 'xlabs_raw':
            print('Next: convert to a standard diffusers LoRA with convert_xflux_lora_to_diffusers.py.')
        elif fmt == 'kontext_raw':
            print('Next: convert_fill_lora.py --mode kontext_raw produces a V2-loadable format directly.')
        elif fmt == 'diffusers':
            print('Next: already a standard diffusers LoRA; check whether it needs compression.')
        else:
            print('Next: unknown format; confirm the key layout by hand.')

        if max(ranks) > 128:
            print('Next: rank > 128; SVD-compress to 128 or lower first, then run visual and metric regression.')
            print('      For example: convert_fill_lora.py --mode diffusers --target-rank 128')
        else:
            print('Next: rank <= 128; it can usually enter V2 dynamic-switching evaluation directly.')


if __name__ == '__main__':
    main()
