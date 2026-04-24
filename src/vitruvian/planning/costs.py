# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""MPPI cost strategies.

A cost strategy scores a batch of candidate terminal embeddings against
a goal embedding. Planner calls ``cost_fn(pred_final, goal)`` and
receives ``(K,)`` costs — the strategy is free to use any distance metric.

Three concrete strategies:

* :class:`MSECost` — shape-agnostic squared-error. Works for flat CLS
  latents ``(K, D)`` and patch latents ``(K, N, D)`` via ``flatten(1)``.
* :class:`PatchMSECost` — per-patch MSE with optional per-patch std
  normalization. Reserved for future use; :class:`MSECost` is the
  default for patch backbones.
* :class:`ValueHeadCost` — squared distance in a learned value-function
  embedding space (VF_quasi, Destrade et al. 2601.00844). Only defined
  for flat 2-D latents.
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn


class CostFn(Protocol):
    """Cost-strategy protocol. ``goal`` is broadcastable over ``pred``."""

    def __call__(
        self, pred: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor: ...


class MSECost:
    """Shape-agnostic squared-error cost.

    Supports both flat ``(K, D)`` and patch ``(K, N, D)`` latents.
    """

    def __call__(
        self, pred: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        diff = pred - goal.to(pred.device, pred.dtype)
        return diff.pow(2).flatten(1).sum(dim=-1)


class PatchMSECost:
    """Per-patch MSE with optional patch-std normalization.

    Args:
        patch_std: Optional ``(N, D)`` tensor of per-patch stds; the
            difference is divided by ``(patch_std + eps)`` before
            squaring. Useful for down-weighting patches that have
            saturated DINOv3 activations.
    """

    def __init__(
        self,
        patch_std: torch.Tensor | None = None,
        *,
        eps: float = 1e-3,
    ) -> None:
        self.patch_std = (
            patch_std.detach().clone() if patch_std is not None else None
        )
        self.eps = float(eps)

    def __call__(
        self, pred: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        assert pred.ndim == 3, (
            f"PatchMSECost expects (K, N, D), got {tuple(pred.shape)}"
        )
        diff = pred - goal.to(pred.device, pred.dtype)
        if self.patch_std is not None:
            std = self.patch_std.to(pred.device, pred.dtype) + self.eps
            diff = diff / std
        return diff.pow(2).flatten(1).sum(dim=-1)


class ValueHeadCost:
    """VF_quasi cost: squared distance in a learned value embedding."""

    def __init__(self, value_head: nn.Module) -> None:
        self.value_head = value_head

    def __call__(
        self, pred: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        if pred.ndim != 2:
            raise RuntimeError(
                f"ValueHeadCost expects (K, D), got {tuple(pred.shape)}. "
                "Patch-latent VF head is not implemented."
            )
        f_pred = self.value_head.f(pred)  # (K, d_v)
        f_goal = self.value_head.f(goal.to(pred.device, pred.dtype).unsqueeze(0))
        return ((f_pred - f_goal) ** 2).sum(dim=-1)


__all__ = ["CostFn", "MSECost", "PatchMSECost", "ValueHeadCost"]
