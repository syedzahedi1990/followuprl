"""Tiny text-hashed policy for engineering checks, never scientific evidence."""

from __future__ import annotations

import hashlib

import torch
from torch import nn


class TinyPolicy(nn.Module):
    def __init__(self, seed: int = 0, features: int = 16):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(features, 8, generator=generator) * 0.05)
        self.bias = nn.Parameter(torch.zeros(8))
        self.features = features
        self.metadata = {
            "model_id": "test-only-text-hash-linear-policy",
            "revision": "builtin-v1",
            "tokenizer_revision": "none",
            "test_only": True,
            "precision": "float32",
            "action_strings": ["0", "1", "2", "3", "4", "5", "?", "!"],
        }

    def action_logits(self, prompts: list[str]) -> torch.Tensor:
        vectors = []
        for text in prompts:
            raw = hashlib.shake_256(text.encode()).digest(self.features)
            vectors.append([(v - 127.5) / 127.5 for v in raw])
        data = torch.tensor(vectors, dtype=self.weight.dtype, device=self.weight.device)
        return data @ self.weight + self.bias
