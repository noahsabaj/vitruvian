# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""High-level dynamics predictor for HWM-on-LeWM.

Replaces HWM's Conv2D spatial predictor (``pldm/models/predictors/
conv_predictors.py``) with a causal-transformer MLP operating on flat
CLS latents. Rationale in docs/decisions/008-hwm-planning-layer.md.

Input:  latent  (B, T, D_latent=192)   — LeWM CLS embeddings
        act_emb (B, T, D_act=32)       — macro-action embeddings
Output: next_latent (B, T, 192)         — predicted next-state latents,
                                          residually added to input
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _TransformerBlock(nn.Module):
    """Pre-LN self-attention + feed-forward, causal masked."""

    def __init__(self, hidden: int, n_heads: int, ff_mult: int = 4) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=n_heads,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, ff_mult * hidden),
            nn.GELU(),
            nn.Linear(ff_mult * hidden, hidden),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        # Pre-LN self-attention
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + h
        # Pre-LN FFN
        x = x + self.ff(self.ln2(x))
        return x


class MLPPredictor(nn.Module):
    """Causal transformer predictor over flat CLS latents + action embeddings.

    At each timestep t, predicts a delta Δz_t such that
        z_{t+1} = z_t + Δz_t
    Given history (z_{0:t}, a_{0:t}).
    """

    def __init__(
        self,
        latent_dim: int = 192,
        action_emb_dim: int = 32,
        hidden: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        max_seq_len: int = 16,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_emb_dim = action_emb_dim
        self.hidden = hidden
        self.max_seq_len = max_seq_len

        input_dim = latent_dim + action_emb_dim
        self.input_proj = nn.Linear(input_dim, hidden)

        # Learned positional embedding. Macro-action sequences are
        # short (O(10)), so a simple parameter is fine.
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, hidden))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.blocks = nn.ModuleList(
            [
                _TransformerBlock(hidden=hidden, n_heads=n_heads)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(hidden)
        self.output_proj = nn.Linear(hidden, latent_dim)

        # Small init on output to start near identity prediction.
        nn.init.zeros_(self.output_proj.bias)
        nn.init.normal_(self.output_proj.weight, std=0.01)

    def forward(
        self, latent: torch.Tensor, act_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            latent: (B, T, latent_dim) input latents at each macro step.
            act_emb: (B, T, action_emb_dim) macro-action embeddings.
        Returns:
            next_latent: (B, T, latent_dim) predicted next-step latents.
        """
        B, T, _ = latent.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"sequence length {T} exceeds max_seq_len {self.max_seq_len}; "
                "bump max_seq_len at construction."
            )

        x = torch.cat([latent, act_emb], dim=-1)  # (B, T, latent+act)
        x = self.input_proj(x)  # (B, T, hidden)
        x = x + self.pos_emb[:, :T]

        # Build causal mask: (T, T), position i only attends to ≤ i.
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )

        for block in self.blocks:
            x = block(x, causal_mask)

        x = self.ln_f(x)
        delta = self.output_proj(x)  # (B, T, latent_dim)
        return latent + delta

    def n_params(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if not trainable_only or p.requires_grad
        )
