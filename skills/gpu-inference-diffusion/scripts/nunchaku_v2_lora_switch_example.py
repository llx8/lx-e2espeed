#!/usr/bin/env python3
"""Nunchaku V2 dynamic LoRA switching with an upstream attention hook."""

import os

from nunchaku.lora.flux.v1.lora_flux_v2 import reset_lora_v2, update_lora_params_v2


class NunchakuV2EffectSwitcher:
    def __init__(self, transformer, lora_paths: dict[str, str], set_attention=None):
        self.transformer = transformer
        self.lora_paths = lora_paths
        self.set_attention = set_attention
        self._current_effect = None
        self._current_attention = None

    def _set_attention(self, processor: str):
        if self._current_attention == processor:
            return
        if self.set_attention is None:
            raise RuntimeError(
                "Attention switching requires a callable backed by the installed "
                "Nunchaku version's native attention API."
            )
        self.set_attention(self.transformer, processor)
        self._current_attention = processor

    def _effect_attention(self, effect: str) -> str:
        return 'sage' if effect == 'fill' else 'flashattn2'

    def switch_effect(self, effect: str, strength: float = 1.0):
        self._set_attention(self._effect_attention(effect))
        if self._current_effect == effect:
            return

        lora_path = self.lora_paths[effect]
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f'LoRA not found: {lora_path}')

        reset_lora_v2(self.transformer)
        update_lora_params_v2(self.transformer, lora_path, strength=strength)
        self._current_effect = effect


if __name__ == '__main__':
    print('Provide set_attention from the installed Nunchaku native API.')
