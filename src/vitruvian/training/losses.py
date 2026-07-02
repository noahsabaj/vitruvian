# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""JEPA training losses.

* :func:`prediction_loss` — 1-step teacher-forced MSE + k-step rollout
  MSE. Shape-polymorphic: accepts batches with either a flat ``"emb"``
  field (v4 CLS) or a ``"patches"`` field (v5, projected internally).
  The prediction **target is a clean, visual-only future embedding**;
  proprioception and action enter as predictor *conditioning*, never
  summed into the target (see :func:`_conditioning`).
* :func:`sigreg_loss` — SIGReg isotropic-Gaussian regularizer (LeJEPA)
  over the projected embedding; the default anti-collapse term.
* :func:`vicreg_std_loss` — the older VICReg variance regularizer, kept
  for ablation / backward reference.

**Design note — proprio as conditioning, not target fusion (M5).**
Recent action-conditioned JEPA world models (V-JEPA 2-AC arXiv:2506.09985,
VLA-JEPA arXiv:2602.10098, Causal-JEPA arXiv:2602.11389) keep the
prediction target a pure future-*state* embedding and inject action /
proprio only as predictor conditioning; Causal-JEPA ablates that
separate conditioning beats fusing auxiliaries into the state latent.
This module follows that: the target is visual-only, so the planner's
goal is a plain image embedding and the MPPI cost is computed in one
clean space. Proprio conditioning is applied with per-frame dropout so
the model stays robust when proprio is absent — which is the case for
the *future* steps of a plan-time rollout, where no proprio is observed.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch
from torch import nn


class _JEPALike(Protocol):
    """Structural protocol for the duck-typed JEPA models this module
    consumes (the unified ``JEPA`` + any shim of the same shape)."""

    proprio_encoder: nn.Module | None
    patch_projector: nn.Module | None
    action_encoder: nn.Module

    def predict(self, emb: torch.Tensor, cond: torch.Tensor) -> torch.Tensor: ...

    def predict_prefix(
        self,
        anchor: torch.Tensor,
        act_emb: torch.Tensor,
        state_cond: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...


def vicreg_std_loss(emb: torch.Tensor, *, eps: float = 1e-4) -> torch.Tensor:
    """Penalize per-channel embedding std below 1 (VICReg variance term).

    ``emb`` is flattened over all non-channel dims before std is
    computed, so this works uniformly for flat ``(B, T, D)`` or patch
    ``(B, T, N, D)`` latents. Kept for ablation; :func:`sigreg_loss` is
    the default (it also suppresses higher-order structure, not just
    per-channel variance).
    """
    flat = emb.reshape(-1, emb.shape[-1])
    std = flat.std(dim=0, unbiased=False) + eps
    return torch.relu(1.0 - std).mean()


def sigreg_loss(
    emb: torch.Tensor,
    *,
    num_proj: int = 1024,
    knots: int = 17,
) -> torch.Tensor:
    """SIGReg isotropic-Gaussian regularizer (LeJEPA, arXiv:2511.08544).

    Pushes the embedding distribution toward an isotropic Gaussian by
    matching its empirical characteristic function to that of ``N(0, I)``
    along ``num_proj`` random 1-D projections, integrated over ``knots``
    Gauss-windowed quadrature points on ``[0, 3]``. It *strictly
    generalizes* the VICReg std term (which only matches second-order
    per-channel variance) and — proven in "When Does LeJEPA Learn a
    World Model?" (arXiv:2605.26379) — yields linearly identifiable
    latents, the property that makes latent-space planning well-posed.
    It needs no EMA / stop-gradient, so it fits the frozen-prior design.

    Every non-channel dim is flattened into the sample axis. We omit
    SIGReg's ``* n`` test-statistic scaling so the term is O(1) (and
    ``reg_weight`` stays comparable to the old VICReg weight), and we
    loop over knots to cap peak memory at ``O(S * num_proj)`` instead of
    materializing an ``(S, num_proj, knots)`` tensor. The random
    projections are resampled each call (a stochastic sketch), matching
    the vendored :class:`vitruvian.lewm_compat.SIGReg`.
    """
    flat = emb.reshape(-1, emb.shape[-1]).float()  # (S, D)
    d = flat.shape[-1]
    dev = flat.device
    a = torch.randn(d, num_proj, device=dev)
    a = a / (a.norm(dim=0, keepdim=True) + 1e-8)
    proj = flat @ a  # (S, num_proj)

    t = torch.linspace(0.0, 3.0, knots, device=dev)
    dt = 3.0 / (knots - 1)
    quad = torch.full((knots,), 2.0 * dt, device=dev)
    quad[0] = dt
    quad[-1] = dt
    phi = torch.exp(-t.square() / 2.0)  # real CF of N(0,1); also the window
    quad = quad * phi

    stat = torch.zeros(num_proj, device=dev)
    for k in range(knots):
        ang = proj * t[k]
        err_k = (ang.cos().mean(0) - phi[k]).square() + ang.sin().mean(0).square()
        stat = stat + quad[k] * err_k
    return stat.mean()


def _compute_emb(
    model: _JEPALike, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the visual-only per-frame embedding the predictor consumes
    (and predicts against).

    Returns ``(emb, emb_for_reg)``. Proprio is NOT fused in — it is a
    conditioning signal (see :func:`_conditioning`), so the target and
    the regularized tensor are both pure visual. The tuple is kept for
    call-site stability; the two entries are currently identical.
    """
    # v4: batch already contains precomputed CLS embeddings.
    if "emb" in batch:
        emb = batch["emb"]  # (B, T, D)
        return emb, emb

    # v5: batch contains precomputed patch tensors; project here.
    if "patches" in batch:
        raw = batch["patches"].float()  # (B, T, N, 768)
        if model.patch_projector is None:
            raise AttributeError(
                "patches batch requires the JEPA to carry a patch_projector"
            )
        emb = model.patch_projector(raw)  # (B, T, N, hidden)
        return emb, emb

    raise KeyError(
        "batch must contain either 'emb' (v4 CLS) or 'patches' (v5)"
    )


def _conditioning(
    model: _JEPALike,
    batch: dict[str, torch.Tensor],
    proprio_dropout: float,
) -> torch.Tensor:
    """Per-frame predictor conditioning ``(B, T, hidden)``.

    ``cond = action_encoder(action) + proprio_encoder(proprio)``. During
    training, whole frames of proprio conditioning are dropped with
    probability ``proprio_dropout`` by zeroing the proprio encoder's
    *output*, so a dropped frame contributes exactly nothing to ``cond``.
    That matches the plan-time rollout regime precisely: there, proprio is
    unobserved and the rollout adds no proprio term at all (see
    :meth:`vitruvian.models.jepa.JEPA.rollout`). Masking the encoder
    *input* instead would leave the learned constant ``proprio_encoder(0)``
    (which is ``!= 0`` — the MLP has biases) in the sum on dropped frames,
    a conditioning offset the plan-time rollout never reproduces.
    """
    cond: torch.Tensor = model.action_encoder(batch["action"])
    pe = model.proprio_encoder
    if pe is not None and "proprio" in batch:
        prop_emb = pe(batch["proprio"].float())  # (B, T, hidden)
        if proprio_dropout > 0.0 and getattr(model, "training", False):
            keep = (
                torch.rand(
                    prop_emb.shape[0], prop_emb.shape[1], 1, device=prop_emb.device
                )
                >= proprio_dropout
            ).to(prop_emb.dtype)
            prop_emb = prop_emb * keep
        cond = cond + prop_emb
    return cond


def prediction_loss(
    model: _JEPALike,
    batch: dict[str, torch.Tensor],
    *,
    history_size: int,
    num_preds: int,
    rollout_weight: float = 1.0,
    reg_weight: float = 0.0,
    proprio_dropout: float = 0.0,
) -> dict[str, Any]:
    """Terver-recipe training loss for any JEPA shape.

    Recipe (Terver et al. arXiv:2512.24497 eq. 5):

    * **1-step teacher-forced MSE** — given context frames ``[0..T_hist-1]``
      and their conditioning, the predictor outputs a per-position
      prediction. Under causal masking, output at position ``t`` depends
      on inputs ``[0..t]``, and is trained against the next frame
      ``emb[t+1]``. So the target is ``emb[:, 1:T_hist+1]``.
    * **k-step rollout MSE** — k from 1 up to ``num_preds - 1``. The
      rolling window advances one step per iteration: drop the oldest
      frame, append the last prediction, recompute. After ``k`` steps the
      window spans frames ``[k .. T_hist + k - 1]`` (its last position is
      frame ``T_hist + k - 1``), so its last-position output predicts the
      *next* frame ``T_hist + k`` — the target ``emb[T_hist + k]``. (For
      the final ``k = num_preds - 1`` this is the last supplied frame, so
      every frame is used.)

    The target ``emb`` is **visual-only**; proprio/action are conditioning
    (see module docstring + :func:`_conditioning`), so the planner's goal
    is a plain image embedding and the MPPI cost lives in one clean space.

    Args:
        model: :class:`vitruvian.models.JEPA` or a duck-typed equivalent
            exposing ``.proprio_encoder``, ``.action_encoder``,
            ``.patch_projector`` and ``.predict(emb, cond)``.
        batch: dict with ``"action"``, optional ``"proprio"``, and either
            ``"emb"`` (v4) or ``"patches"`` (v5).
        history_size: ``T_hist``; the predictor consumes the first
            ``T_hist`` frames.
        num_preds: max rollout horizon ``K``; sequence length must be
            ``T_hist + K``.
        rollout_weight: coefficient on the k-step rollout MSE.
        reg_weight: coefficient on :func:`sigreg_loss` over the projected
            embedding. Use a positive value (e.g. 1.0) with the v5 patch
            head to prevent collapse. (Renamed from ``std_weight``; it now
            weights SIGReg, not the VICReg std term.)
        proprio_dropout: probability of zeroing a frame's proprio
            conditioning during training (0 disables). ~0.5 keeps the
            model robust to the proprio-free future of a plan rollout.

    Returns:
        Dict with ``"loss"`` (combined), ``"pred_loss"``,
        ``"rollout_loss"``, ``"reg_loss"``, ``"n_rollout_steps"``.
    """
    emb, emb_for_reg = _compute_emb(model, batch)
    cond = _conditioning(model, batch, proprio_dropout)

    ctx_len = int(history_size)
    n_preds = int(num_preds)

    ctx_emb = emb[:, :ctx_len]
    ctx_cond = cond[:, :ctx_len]
    # 1-step TF target: at each context position t, predict emb[t+1].
    tgt_emb = emb[:, 1 : ctx_len + 1]

    pred_emb = model.predict(ctx_emb, ctx_cond)
    pred_loss = (pred_emb - tgt_emb).pow(2).mean()

    rollout_losses: list[torch.Tensor] = []
    if rollout_weight > 0 and n_preds >= 2:
        rolling_emb = ctx_emb
        for k in range(1, n_preds):
            rolling_emb = torch.cat(
                [rolling_emb[:, 1:], pred_emb[:, -1:]], dim=1
            )
            rolling_cond = cond[:, k : k + ctx_len]
            pred_emb = model.predict(rolling_emb, rolling_cond)
            # Window ends at frame ``k + ctx_len - 1``; its last-position
            # output predicts the NEXT frame, so the target is
            # ``emb[k + ctx_len]`` (not ``k + ctx_len - 1``, which is the
            # window's own last input and would train a copy).
            tgt_k = emb[:, k + ctx_len : k + ctx_len + 1]
            rollout_losses.append((pred_emb[:, -1:] - tgt_k).pow(2).mean())
        rollout_loss = torch.stack(rollout_losses).mean()
    else:
        rollout_loss = torch.zeros((), device=emb.device)

    reg_loss = (
        sigreg_loss(emb_for_reg)
        if reg_weight > 0
        else torch.zeros((), device=emb.device)
    )

    total = pred_loss + rollout_weight * rollout_loss + reg_weight * reg_loss
    return {
        "loss": total,
        "pred_loss": pred_loss.detach(),
        "rollout_loss": rollout_loss.detach(),
        "reg_loss": reg_loss.detach(),
        "n_rollout_steps": len(rollout_losses),
    }


def prefix_prediction_loss(
    model: _JEPALike,
    batch: dict[str, torch.Tensor],
    *,
    history_size: int,
    num_preds: int,
    rollout_weight: float = 1.0,
    reg_weight: float = 0.0,
    proprio_dropout: float = 0.0,
) -> dict[str, Any]:
    """Fast-LeWM (arXiv:2606.26217) dense action-prefix loss.

    The anchor is the last of the ``history_size`` context frames — Fast-LeWM
    anchors *every* prediction on the OBSERVED latent, so ``history_size=1`` is
    the canonical setting. From that single anchor the predictor emits all
    ``num_preds`` future latents in one pass, each conditioned on the action
    *prefix* to that horizon; we supervise every horizon densely against the
    real future frames. No autoregressive chaining → no compounding error (the
    M6 fix for the Q1a horizon-error growth).

    Requires ``model.predict_prefix`` and a patch batch (``"patches"``). The
    anchor's proprio is folded in as ``state_cond``; ``proprio_dropout`` zeroes
    that ``state_cond`` (the encoder *output*) per sample so a dropped anchor
    contributes nothing — identical to the plan-time rollout, which passes
    ``state_cond=None`` because proprio is unobserved there. For logging parity
    with :func:`prediction_loss`, the 1-step term is reported as ``pred_loss``
    and horizons ≥2 as ``rollout_loss``.
    """
    emb, emb_for_reg = _compute_emb(model, batch)  # (B, T, N, hidden)
    if emb.dim() != 4:
        raise ValueError(
            "prefix_prediction_loss needs a patch batch (B, T, N, D)"
        )
    a0 = int(history_size) - 1  # anchor frame index
    k = int(num_preds)
    anchor = emb[:, a0]  # (B, N, hidden)
    targets = emb[:, a0 + 1 : a0 + 1 + k]  # (B, k, N, hidden)
    actions = batch["action"][:, a0 : a0 + k]  # (B, k, A): drive anchor -> +k
    act_emb = model.action_encoder(actions)  # (B, k, hidden)

    state_cond: torch.Tensor | None = None
    pe = model.proprio_encoder
    if pe is not None and "proprio" in batch:
        prop0 = batch["proprio"][:, a0].float()  # (B, P): anchor proprio
        state_cond = pe(prop0)  # (B, hidden)
        if proprio_dropout > 0.0 and getattr(model, "training", False):
            # Zero the encoder OUTPUT (not the input) so a dropped anchor adds
            # nothing — matching the plan-time ``state_cond=None`` regime.
            keep = (
                torch.rand(state_cond.shape[0], 1, device=state_cond.device)
                >= proprio_dropout
            ).to(state_cond.dtype)
            state_cond = state_cond * keep

    pred = model.predict_prefix(anchor, act_emb, state_cond)  # (B, k, N, hidden)

    pred_loss = (pred[:, :1] - targets[:, :1]).pow(2).mean()
    has_rollout = rollout_weight > 0 and k >= 2
    if has_rollout:
        rollout_loss = (pred[:, 1:] - targets[:, 1:]).pow(2).mean()
    else:
        rollout_loss = torch.zeros((), device=emb.device)
    reg_loss = (
        sigreg_loss(emb_for_reg)
        if reg_weight > 0
        else torch.zeros((), device=emb.device)
    )
    total = pred_loss + rollout_weight * rollout_loss + reg_weight * reg_loss
    return {
        "loss": total,
        "pred_loss": pred_loss.detach(),
        "rollout_loss": rollout_loss.detach(),
        "reg_loss": reg_loss.detach(),
        # Number of supervised rollout horizons (0 when the rollout term is
        # off), matching prediction_loss's ``len(rollout_losses)`` semantics.
        "n_rollout_steps": (k - 1) if has_rollout else 0,
    }


__all__ = [
    "prediction_loss",
    "prefix_prediction_loss",
    "sigreg_loss",
    "vicreg_std_loss",
]
