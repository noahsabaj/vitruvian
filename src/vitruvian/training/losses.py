# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""JEPA training losses.

* :func:`prediction_loss` — 1-step teacher-forced MSE + k-step rollout
  MSE. Shape-polymorphic: accepts batches with either a flat
  ``"emb"`` field (v4 CLS) or a ``"patches"`` field (v5, projected
  internally).
* :func:`vicreg_std_loss` — VICReg variance regularizer over the
  projected embedding. Prevents ``patch_projector`` from collapsing
  to a constant — the empirical failure mode of the v5 first pass.
"""

from __future__ import annotations

import torch
from torch import nn


def vicreg_std_loss(emb: torch.Tensor, *, eps: float = 1e-4) -> torch.Tensor:
    """Penalize per-channel embedding std below 1.

    ``emb`` is flattened over all non-channel dims before std is
    computed, so this works uniformly for flat ``(B, T, D)`` or patch
    ``(B, T, N, D)`` latents.
    """
    flat = emb.reshape(-1, emb.shape[-1])
    std = flat.std(dim=0, unbiased=False) + eps
    return torch.relu(1.0 - std).mean()


def _fuse_proprio(
    emb: torch.Tensor, proprio: torch.Tensor, proprio_encoder: nn.Module | None
) -> torch.Tensor:
    if proprio_encoder is None:
        return emb
    prop_emb = proprio_encoder(proprio.float())  # (B, T, hidden)
    if emb.dim() == 4:
        return emb + prop_emb.unsqueeze(2)
    return emb + prop_emb


def _compute_emb(
    model: nn.Module, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the per-frame embedding the predictor will consume.

    Returns ``(emb, emb_for_std)`` — the input to the predictor and a
    separate tensor for the VICReg std regularizer. For v4 both are
    identical; for v5 the std is computed AFTER projection + proprio
    fusion (where collapse actually happens).
    """
    # v4: batch already contains precomputed CLS embeddings.
    if "emb" in batch:
        emb = batch["emb"]  # (B, T, D)
        emb = _fuse_proprio(emb, batch["proprio"], model.proprio_encoder)
        return emb, emb

    # v5: batch contains precomputed patch tensors; project + fuse here.
    if "patches" in batch:
        raw = batch["patches"].float()  # (B, T, N, 768)
        if model.patch_projector is None:
            raise AttributeError(
                "patches batch requires the JEPA to carry a patch_projector"
            )
        emb = model.patch_projector(raw)  # (B, T, N, hidden)
        emb = _fuse_proprio(emb, batch["proprio"], model.proprio_encoder)
        return emb, emb

    raise KeyError(
        "batch must contain either 'emb' (v4 CLS) or 'patches' (v5)"
    )


def prediction_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    history_size: int,
    num_preds: int,
    rollout_weight: float = 1.0,
    std_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Terver-recipe training loss for any JEPA shape.

    Args:
        model: :class:`vitruvian.models.JEPA` or a legacy ``JEPAv4`` /
            ``JEPAv5`` — duck-typed to ``.proprio_encoder``,
            ``.action_encoder``, ``.patch_projector`` / ``.patch_proj``,
            and ``.predict(emb, act_emb)``.
        batch: dict with ``"proprio"``, ``"action"``, and either
            ``"emb"`` (v4) or ``"patches"`` (v5).
        history_size: ``T_hist``; the predictor consumes the first
            ``T_hist`` frames.
        num_preds: max rollout horizon ``K``; total sequence length
            must be ``T_hist + K``.
        rollout_weight: coefficient on the k-step rollout MSE.
        std_weight: coefficient on :func:`vicreg_std_loss` over the
            projected embedding. Use a positive value (e.g. 1.0) with
            the v5 patch head to prevent collapse.

    Returns:
        Dict with ``"loss"`` (combined), ``"pred_loss"``,
        ``"rollout_loss"``, ``"std_loss"``, ``"n_rollout_steps"``.
    """
    emb, emb_for_std = _compute_emb(model, batch)

    act_emb = model.action_encoder(batch["action"])

    ctx_len = int(history_size)
    n_preds = int(num_preds)

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds : n_preds + ctx_len]

    pred_emb = model.predict(ctx_emb, ctx_act)
    pred_loss = (pred_emb - tgt_emb).pow(2).mean()

    rollout_losses: list[torch.Tensor] = []
    if rollout_weight > 0 and n_preds >= 2:
        rolling_emb = ctx_emb
        for k in range(1, n_preds):
            rolling_emb = torch.cat(
                [rolling_emb[:, 1:], pred_emb[:, -1:]], dim=1
            )
            rolling_act = act_emb[:, k : k + ctx_len]
            pred_emb = model.predict(rolling_emb, rolling_act)
            tgt_k = emb[:, k + ctx_len - 1 : k + ctx_len]
            rollout_losses.append((pred_emb[:, -1:] - tgt_k).pow(2).mean())
        rollout_loss = sum(rollout_losses) / max(1, len(rollout_losses))
    else:
        rollout_loss = torch.zeros((), device=emb.device)

    std_loss = (
        vicreg_std_loss(emb_for_std)
        if std_weight > 0
        else torch.zeros((), device=emb.device)
    )

    total = pred_loss + rollout_weight * rollout_loss + std_weight * std_loss
    return {
        "loss": total,
        "pred_loss": pred_loss.detach(),
        "rollout_loss": (
            rollout_loss.detach()
            if torch.is_tensor(rollout_loss)
            else torch.zeros((), device=emb.device)
        ),
        "std_loss": std_loss.detach(),
        "n_rollout_steps": len(rollout_losses),
    }


__all__ = ["prediction_loss", "vicreg_std_loss"]
