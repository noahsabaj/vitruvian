# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""High-level world model composition: frozen LeWM encoder +
trainable macro-action encoder + trainable MLP predictor.

This is what we train in M4.4b and plan against in M4.4c.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .action_codec import MacroActionEncoder
from .backbone_adapter import LeWMBackboneAdapter
from .mlp_predictor import MLPPredictor


class HighLevelModel(nn.Module):
    """Trainable high-level dynamics model on top of a frozen LeWM.

    Forward signatures:
        encode(pixels)  -> (B, T, D_latent)       [no-grad, via adapter]
        predict(emb, macro_actions) -> (B, T-1, D) delta-style prediction
                                                  of z_{t+1} from z_t, a_t

    Only ``action_encoder`` and ``predictor`` receive gradients.
    """

    def __init__(
        self,
        lewm_jepa: nn.Module,
        *,
        action_dim: int = 29,
        step_skip: int = 50,
        macro_act_dim: int = 32,
        hidden: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        max_seq_len: int = 16,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = LeWMBackboneAdapter(lewm_jepa, freeze=freeze_backbone)
        self.action_encoder = MacroActionEncoder(
            action_dim=action_dim,
            step_skip=step_skip,
            macro_act_dim=macro_act_dim,
        )
        self.predictor = MLPPredictor(
            latent_dim=self.backbone.output_dim,
            action_emb_dim=macro_act_dim,
            hidden=hidden,
            n_layers=n_layers,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
        )

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) -> (B, T, D_latent). No gradient through backbone."""
        return self.backbone.encode(pixels)

    def predict_next_latents(
        self,
        emb: torch.Tensor,            # (B, T+1, D_latent)
        macro_actions: torch.Tensor,  # (B, T, step_skip, action_dim)
    ) -> torch.Tensor:
        """Predict latents at macro steps 1..T given latents 0..T and actions 0..T-1.

        Args:
            emb: (B, T+1, D_latent) encoder outputs at each macro boundary.
            macro_actions: (B, T, step_skip, action_dim) primitive-action
                chunks for each macro.
        Returns:
            predicted_next: (B, T, D_latent) — one prediction per macro.
        """
        macro_emb = self.action_encoder(macro_actions)      # (B, T, macro_act_dim)
        # Condition predictor on latents at 0..T-1 and actions 0..T-1,
        # producing predictions for 1..T.
        return self.predictor(emb[:, :-1], macro_emb)

    def trainable_parameters(self):
        for p in self.parameters():
            if p.requires_grad:
                yield p

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())
