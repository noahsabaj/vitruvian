# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""HWM-on-LeWM training objectives.

Two losses applied to the high-level world model:
    - PredictionLoss: teacher-forcing MSE between predicted next-latent
      and target next-latent (structure from
      external/hwm/pldm/objectives/prediction.py).
    - VICRegLoss: variance + covariance regularizer on the predicted
      latents (structure from external/hwm/pldm/objectives/vicreg.py);
      similarity term dropped because the prediction loss already
      enforces similarity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossInfo:
    total_loss: torch.Tensor
    components: dict  # str -> torch.Tensor (unweighted, for logging)


class PredictionLoss(nn.Module):
    """Teacher-forcing MSE between predicted and target latents.

    For each macro timestep t in 0..T-2, the prediction at t is
    compared against the encoded target at t+1.
    """

    def __init__(self, coeff: float = 1.0) -> None:
        super().__init__()
        self.coeff = coeff

    def forward(
        self, predicted: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predicted: (B, T, D) predictions at macro steps 0..T-1
                (each step is the one-step-ahead prediction).
            target:    (B, T, D) ground-truth latents at macro steps
                1..T (shifted by one step).
        Returns:
            scalar MSE loss.
        """
        if predicted.shape != target.shape:
            raise ValueError(
                f"shape mismatch: predicted {tuple(predicted.shape)} vs "
                f"target {tuple(target.shape)}"
            )
        return F.mse_loss(predicted, target) * self.coeff


class VICRegLoss(nn.Module):
    """Variance + covariance regularizer on predicted latents.

    Skips the similarity (invariance) term because the prediction
    loss (teacher-forcing MSE to the target) already enforces it.
    See external/hwm/pldm/objectives/vicreg.py and Bardes et al.
    arXiv:2105.04906.

    Defaults: std_coeff=25, cov_coeff=1 — matches HWM's L2 config.
    """

    def __init__(
        self,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.eps = eps

    def forward(self, emb: torch.Tensor) -> dict:
        """
        Args:
            emb: (B, T, D) latent embeddings over a sequence.
                Batched-time-flat for regularization (B*T treated as
                the "batch" axis).
        Returns:
            dict with 'total', 'std_loss', 'cov_loss' tensors.
        """
        B, T, D = emb.shape
        x = emb.reshape(B * T, D)  # treat each (b, t) as a sample

        # Variance term: penalize per-dim stddev for falling below 1.
        std = torch.sqrt(x.var(dim=0) + self.eps)  # (D,)
        std_loss = torch.mean(F.relu(1.0 - std))

        # Covariance term: penalize off-diagonal cov entries.
        x_centered = x - x.mean(dim=0, keepdim=True)
        N = x_centered.shape[0]
        # (D, D) covariance matrix
        cov = (x_centered.T @ x_centered) / max(1, N - 1)
        off_diag = cov.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()
        cov_loss = (off_diag.pow(2).sum()) / D

        total = self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return {"total": total, "std_loss": std_loss, "cov_loss": cov_loss}
