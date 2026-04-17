# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Macro-action encoder for HWM-on-LeWM.

Encodes a temporal chunk of primitive actions into a macro-action
latent. Inspired by LeWM's ``module.Embedder`` pattern but operates
over longer temporal windows (step_skip=50 by default, = 1 s at
G1's 50 Hz control rate).

Input:  primitives (B, T_macro, step_skip=50, action_dim=29)
Output: macro_emb  (B, T_macro, macro_act_dim=32)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MacroActionEncoder(nn.Module):
    """1D-conv temporal encoder over primitive actions.

    Three Conv1d(kernel=5, stride=2) stages reduce a length-50
    sequence down to ~6 temporal tokens, which are mean-pooled to
    produce the macro-action latent. Channels progress
    ``action_dim → hidden_1 → hidden_2 → macro_act_dim``.

    Uses GELU nonlinearities and LayerNorm between stages.
    """

    def __init__(
        self,
        action_dim: int = 29,
        step_skip: int = 50,
        hidden_1: int = 64,
        hidden_2: int = 64,
        macro_act_dim: int = 32,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.step_skip = step_skip
        self.macro_act_dim = macro_act_dim

        # input: (B*T_macro, action_dim, step_skip)
        self.conv1 = nn.Conv1d(action_dim, hidden_1, kernel_size=5, stride=2, padding=2)
        self.ln1 = nn.GroupNorm(1, hidden_1)
        self.conv2 = nn.Conv1d(hidden_1, hidden_2, kernel_size=5, stride=2, padding=2)
        self.ln2 = nn.GroupNorm(1, hidden_2)
        self.conv3 = nn.Conv1d(
            hidden_2, macro_act_dim, kernel_size=5, stride=2, padding=2
        )
        self.act = nn.GELU()

    def forward(self, primitives: torch.Tensor) -> torch.Tensor:
        """
        Args:
            primitives: (B, T_macro, step_skip, action_dim) — primitive
                actions grouped into macros.
        Returns:
            macro_emb: (B, T_macro, macro_act_dim)
        """
        B, T, S, A = primitives.shape
        if S != self.step_skip:
            raise ValueError(
                f"expected step_skip={self.step_skip}, got {S}. "
                "Check dataset's step_skip matches encoder's."
            )
        if A != self.action_dim:
            raise ValueError(
                f"expected action_dim={self.action_dim}, got {A}."
            )

        # (B*T, step_skip, action_dim) -> (B*T, action_dim, step_skip) for Conv1d
        x = primitives.reshape(B * T, S, A).transpose(1, 2)

        x = self.act(self.ln1(self.conv1(x)))  # (B*T, hidden_1, S/2)
        x = self.act(self.ln2(self.conv2(x)))  # (B*T, hidden_2, S/4)
        x = self.conv3(x)  # (B*T, macro_act_dim, S/8)

        # mean over remaining temporal dim → (B*T, macro_act_dim)
        x = x.mean(dim=-1)
        return x.view(B, T, self.macro_act_dim)

    def n_params(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if not trainable_only or p.requires_grad
        )
