# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""VF-HER trainer — IQL expectile regression for a goal-conditioned V head.

Trains ``V_ψ(s, g) = -||f_ψ(s) - f_ψ(g)||²`` on frozen JEPAv4 latents
with HER-style future-state goal relabeling. The learned ``f_ψ`` powers
:class:`vitruvian.planning.costs.ValueHeadCost` at plan time.

Hyperparameters (``expectile=0.7``, ``ema=0.001``, ``gamma=0.98``,
``lr=1e-4``) carry over from the stable M4.4d v2 recipe.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueHead(nn.Module):
    """``f_ψ: emb_D -> R^d``; ``V(s, g) = -||f(s) - f(g)||²``.

    The learned ``f`` shapes the distance metric used by the planner.
    The raw embedding space (DINOv3 CLS or LeWM CLS) is pose-invariant;
    ``f`` distorts it into one that tracks goal-reach distance.
    """

    def __init__(
        self, emb_dim: int = 768, hidden: int = 256, out_dim: int = 128
    ) -> None:
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, s_emb: torch.Tensor, g_emb: torch.Tensor) -> torch.Tensor:
        """Returns scalar ``V ≤ 0``."""
        v: torch.Tensor = -((self.f(s_emb) - self.f(g_emb)) ** 2).sum(-1)
        return v


def ema_update(target: nn.Module, source: nn.Module, rate: float) -> None:
    """Polyak-averaging of ``source`` into ``target`` (in-place)."""
    with torch.no_grad():
        for p_t, p_s in zip(target.parameters(), source.parameters()):
            p_t.data.mul_(1.0 - rate).add_(p_s.data, alpha=rate)


def expectile_loss(
    diff: torch.Tensor, expectile: float = 0.7
) -> torch.Tensor:
    """Asymmetric L2 (IQL's ``L2^τ``). ``τ > 0.5`` up-weights positive
    residuals — the value head becomes an upper expectile of TD targets.
    """
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


@dataclass
class VFHERConfig:
    epochs: int = 15
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 1e-4
    expectile: float = 0.7
    gamma: float = 0.98
    ema_rate: float = 0.001
    reward_self_loop: float = 0.0
    reward_step: float = -1.0
    bf16: bool = True
    num_workers: int = 4


class VFHERTrainer:
    """IQL-expectile trainer for the VF head with an EMA target.

    Requires a frozen proprio encoder (from a trained JEPAv4 checkpoint)
    to fuse ``emb_vis`` with proprio into the latent space the VF head
    learns over. The proprio encoder is NOT trained here.
    """

    def __init__(
        self,
        value_head: ValueHead,
        proprio_encoder: nn.Module,
        cfg: VFHERConfig,
    ) -> None:
        self.value_head = value_head
        self.target_head = copy.deepcopy(value_head)
        for p in self.target_head.parameters():
            p.requires_grad_(False)
        self.proprio_encoder = proprio_encoder
        for p in self.proprio_encoder.parameters():
            p.requires_grad_(False)
        self.proprio_encoder.eval()
        self.cfg = cfg

    def fuse(
        self, emb_vis: torch.Tensor, proprio: torch.Tensor
    ) -> torch.Tensor:
        prop_emb: torch.Tensor = self.proprio_encoder(proprio.float())
        return emb_vis + prop_emb

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        s_emb = self.fuse(batch["emb_t"], batch["prop_t"])
        sp_emb = self.fuse(batch["emb_tp1"], batch["prop_tp1"])
        g_emb = self.fuse(batch["emb_g"], batch["prop_g"])
        is_self = batch["is_goal_self"].bool()

        r = torch.where(
            is_self,
            torch.full_like(is_self, self.cfg.reward_self_loop, dtype=s_emb.dtype),
            torch.full_like(is_self, self.cfg.reward_step, dtype=s_emb.dtype),
        )
        with torch.no_grad():
            v_next = self.target_head(sp_emb, g_emb)
        v_now = self.value_head(s_emb, g_emb)
        target = r + self.cfg.gamma * v_next
        loss = expectile_loss(target - v_now, self.cfg.expectile)
        return {
            "loss": loss,
            "v_now_mean": v_now.detach().mean(),
            "v_now_min": v_now.detach().min(),
            "target_mean": target.detach().mean(),
        }

    def ema_step(self) -> None:
        ema_update(self.target_head, self.value_head, self.cfg.ema_rate)


__all__ = ["VFHERConfig", "VFHERTrainer", "ValueHead", "ema_update", "expectile_loss"]
