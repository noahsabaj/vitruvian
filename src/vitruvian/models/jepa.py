# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Unified JEPA composer.

Subsumes the three milestone-specific composers (v3 LeWM, v4 DINOv3-CLS,
v5 DINOv3-patch) into one :class:`JEPA` class that is shape-polymorphic
over the backbone output — a 3-D ``(B, T, D)`` flat latent or a 4-D
``(B, T, N, D)`` patch latent.

The code path per forward:

1. ``backbone.encode(pixels)`` produces ``emb`` (3-D or 4-D).
2. If a :attr:`patch_projector` is attached, it maps the backbone's
   raw per-token dim (e.g. 768 for DINOv3) down to the predictor's
   ``hidden_dim`` (e.g. 256).  No-op for flat-CLS backbones.
3. If a :attr:`proprio_encoder` is attached, it produces a per-frame
   embedding that is broadcast-added into ``emb`` (same scalar broadcast
   as v4; unsqueezes over the patch axis for v5).
4. ``action_encoder`` turns raw actions into per-frame conditioning.
5. ``predictor`` autoregresses over the embedding given per-frame
   conditioning.

The rollout API is identical across configurations so
:class:`vitruvian.planning.mppi.MPPIPlanner` does not care which JEPA
shape is behind it.
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn
from einops import rearrange


def _detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


class _BackboneLike(Protocol):
    output_dim: int

    def encode(self, pixels: torch.Tensor) -> torch.Tensor: ...


class JEPA(nn.Module):
    """Backbone + predictor + optional proprio / patch projectors.

    Args:
        backbone: Frozen visual encoder. Must expose ``encode(pixels)``
            returning either ``(B, T, D)`` or ``(B, T, N, D)``.
        predictor: Autoregressive next-step predictor. Must accept
            ``(x, c)`` with ``x`` matching the embedding rank (3-D or
            4-D) and ``c`` the per-frame conditioning.
        action_encoder: Maps raw actions to per-frame conditioning.
        proprio_encoder: Optional MLP over proprioception. Output is
            broadcast-summed into the embedding (unsqueezed over the
            patch axis for patch latents).
        patch_projector: Optional per-token linear projection applied
            right after the backbone. Required for patch backbones
            whose raw token dim (e.g. 768) differs from the predictor
            ``hidden_dim`` (e.g. 256). None for flat-CLS.
    """

    def __init__(
        self,
        backbone: _BackboneLike,
        predictor: nn.Module,
        action_encoder: nn.Module,
        proprio_encoder: nn.Module | None = None,
        patch_projector: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.proprio_encoder = proprio_encoder
        self.patch_projector = patch_projector

        # Cache a few handy shape props for downstream consumers.
        self.backbone_output_dim = int(getattr(backbone, "output_dim"))
        self.n_patches: int | None = getattr(backbone, "n_patches", None)
        # Predictor's hidden_dim (what the rest of the stack sees).
        self.emb_dim = int(
            getattr(predictor, "hidden_dim", self.backbone_output_dim)
        )

    # ------------------------------------------------------------------
    # Encode / predict / rollout
    # ------------------------------------------------------------------

    def _project(self, emb_raw: torch.Tensor) -> torch.Tensor:
        if self.patch_projector is None:
            return emb_raw
        return self.patch_projector(emb_raw)

    def _fuse_proprio(
        self, emb: torch.Tensor, proprio: torch.Tensor
    ) -> torch.Tensor:
        if self.proprio_encoder is None:
            return emb
        prop_emb = self.proprio_encoder(proprio.float())  # (B, T, hidden)
        if emb.dim() == 4:  # patch latents: broadcast over N
            return emb + prop_emb.unsqueeze(2)
        return emb + prop_emb  # flat CLS: direct sum

    def encode(self, info: dict) -> dict:
        """Encode pixels (+ optional proprio) to the JEPA latent.

        ``info`` must contain ``"pixels": (B, T, 3, H, W)``. If
        ``"proprio"`` is present and a proprio encoder is attached, it
        fuses into the embedding. If ``"action"`` is present it is
        conditioned through ``action_encoder``.

        Writes ``info["emb"]`` (shape matches backbone output rank) and
        ``info["act_emb"]`` if action present. Returns ``info``.
        """
        pixels = info["pixels"]
        emb_raw = self.backbone.encode(pixels)
        emb = self._project(emb_raw)
        if "proprio" in info:
            emb = self._fuse_proprio(emb, info["proprio"])
        info["emb"] = emb
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """Delegate to the underlying predictor. Accepts both 3-D and
        4-D embeddings; the predictor itself imposes the shape check.
        """
        return self.predictor(emb, act_emb)

    def rollout(
        self,
        info: dict,
        action_sequence: torch.Tensor,
        history_size: int = 3,
    ) -> dict:
        """Autoregressive latent rollout for MPPI candidate scoring.

        Accepts EITHER pre-encoded ``info["emb"]`` OR raw
        ``info["pixels"]``. Pre-encoded is strictly preferred at plan
        time — the caller (typically ``EncoderHistory``) amortizes the
        backbone forward across MPPI iterations.

        Shape conventions (both flat and patch layouts supported):

            info["emb"]        : (B, S, T_hist, ...), where the trailing
                                 dims are ``(D,)`` for flat CLS or
                                 ``(N, D)`` for patches.
            info["pixels"]     : (B, S, T_hist, 3, H, W)
            action_sequence    : (B, S, T, action_dim * frameskip),
                                 with ``T = T_hist + n_future``.

        Writes ``info["predicted_emb"]`` with shape
        ``(B, S, T_total, ...)``.
        """
        if "emb" in info:
            emb_init = info["emb"]
            H = emb_init.size(2)
        else:
            assert "pixels" in info, (
                "rollout() needs either info['emb'] (pre-encoded) or "
                "info['pixels']"
            )
            H = info["pixels"].size(2)

        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        if "emb" in info:
            emb = info["emb"] = emb_init
        else:
            _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            _init = self.encode(_init)
            emb_single = _init["emb"]  # (B, H, ...)
            # Expand over S samples.
            if emb_single.dim() == 3:  # (B, T, D)
                emb = info["emb"] = emb_single.unsqueeze(1).expand(B, S, -1, -1)
            else:  # (B, T, N, D)
                emb = info["emb"] = (
                    emb_single.unsqueeze(1).expand(B, S, -1, -1, -1)
                )
            _init = {k: _detach_clone(v) for k, v in _init.items()}

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]
            act_trunc = act_emb[:, -HS:]
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)

            next_act = act_future[:, t : t + 1, :]
            act = torch.cat([act, next_act], dim=1)

        # Final step.
        act_emb = self.action_encoder(act)
        emb_trunc = emb[:, -HS:]
        act_trunc = act_emb[:, -HS:]
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)

        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout
        return info


class PlannerBackbone(nn.Module):
    """Planner-facing wrapper exposing the ``encode()`` path **including**
    the optional patch projector but **excluding** proprio.

    :class:`JEPA.encode` bakes proprio into the embedding when present,
    but the planner's :class:`EncoderHistory` encodes from pixels alone.
    This wrapper applies ``backbone.encode`` → (optional)
    ``patch_projector`` so the history buffer produces the same
    projected embedding the predictor consumes during rollout.
    """

    def __init__(self, jepa: JEPA) -> None:
        super().__init__()
        self.jepa = jepa
        self.output_dim = jepa.emb_dim
        self.n_patches = jepa.n_patches

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        raw = self.jepa.backbone.encode(pixels)
        if self.jepa.patch_projector is not None:
            return self.jepa.patch_projector(raw)
        return raw

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.encode(pixels)


__all__ = ["JEPA", "PlannerBackbone"]
