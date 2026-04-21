# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""M4.6 — JEPAv5: frozen DINOv3 patch backbone + trainable patch projector
+ proprio MLP + ``PatchARPredictor`` + LeWM action ``Embedder``.

Composes the M4.6 modules. Exposes the same
``encode`` / ``predict`` / ``rollout`` API as ``JEPAv4`` so
``LowLevelPlanner`` in ``src/vitruvian/hwm/planners.py`` needs no
changes once its cost function is made shape-agnostic.

Design differences from v4:

- **Encoder** returns ``(B, T, 49, 768)`` patches instead of
  ``(B, T, 768)`` CLS.
- **Projector** is now a *trainable* ``Linear(768 → 256)`` applied
  per-patch, reducing the predictor's hidden dim to 256 for the 8 GB
  VRAM budget.
- **Proprio MLP** outputs 256-D and is **broadcast-added** to every
  patch in the frame (not summed with a single CLS).
- **Predictor** is ``PatchARPredictor`` (factorized spatial+temporal
  attention), replacing LeWM's temporal-only ``ARPredictor``.
- **Action encoder** still uses LeWM's ``Embedder`` but at
  ``emb_dim=256`` to match the predictor's hidden dim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange

# Pull LeWM's Embedder + MLP without needing stable_pretraining.
_ROOT = Path(__file__).resolve().parents[3]
_LEWM_SRC = _ROOT / "external" / "le-wm"
if str(_LEWM_SRC) not in sys.path:
    sys.path.insert(0, str(_LEWM_SRC))

import module as lewm_module  # type: ignore[import-not-found]  # noqa: E402

from vitruvian.hwm.backbone_dinov3_patches import (  # noqa: E402
    DEFAULT_DINOV3_ID,
    DINOv3PatchBackbone,
)
from vitruvian.hwm.patch_predictor import PatchARPredictor  # noqa: E402


def _detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


class JEPAv5(nn.Module):
    """DINOv3-patch-backed world model, LowLevelPlanner-compatible."""

    def __init__(
        self,
        *,
        dinov3_model_id: str = DEFAULT_DINOV3_ID,
        spatial_stride: int = 2,  # 14×14 -> 7×7 patches
        proprio_dim: int | None = 103,
        proprio_hidden: int = 256,
        action_dim: int = 29,
        action_frameskip: int = 1,
        action_smoothed_dim: int = 10,
        predictor_num_frames: int = 3,
        predictor_depth: int = 6,
        predictor_heads: int = 8,
        predictor_mlp_dim: int = 1024,
        predictor_hidden: int = 256,
        predictor_dim_head: int = 32,
        predictor_dropout: float = 0.1,
        predictor_adaln_rank: int = 128,
        device: str = "cuda",
        dinov3_dtype: torch.dtype = torch.float16,
        backbone_lazy: bool = False,
    ) -> None:
        super().__init__()

        # Visual (frozen). ``backbone_lazy=True`` skips the 344 MB DINOv3
        # weight load — useful when training reads a precomputed patch
        # cache. The backbone hydrates on first ``encode()`` call.
        self.backbone = DINOv3PatchBackbone(
            model_id=dinov3_model_id,
            device=device,
            dtype=dinov3_dtype,
            freeze=True,
            spatial_stride=spatial_stride,
            lazy=backbone_lazy,
        )
        patch_dim = self.backbone.output_dim  # 768 for ViT-B/16
        n_patches = self.backbone.n_patches    # 49 for stride=2

        # Trainable per-patch projector: 768 -> 256.
        self.patch_proj = nn.Linear(patch_dim, predictor_hidden).to(device)

        # Trainable proprio branch — output broadcast-added to every patch.
        if proprio_dim is not None and proprio_dim > 0:
            self.proprio_encoder = lewm_module.MLP(
                input_dim=int(proprio_dim),
                output_dim=predictor_hidden,
                hidden_dim=proprio_hidden,
                norm_fn=None,
            ).to(device)
        else:
            self.proprio_encoder = None

        # Action encoder: Embedder at emb_dim=predictor_hidden for AdaLN.
        self.action_encoder = lewm_module.Embedder(
            input_dim=int(action_dim) * int(action_frameskip),
            smoothed_dim=int(action_smoothed_dim),
            emb_dim=predictor_hidden,
        ).to(device)

        # Predictor.
        self.predictor = PatchARPredictor(
            num_frames=predictor_num_frames,
            num_patches=n_patches,
            depth=predictor_depth,
            heads=predictor_heads,
            mlp_dim=predictor_mlp_dim,
            input_dim=predictor_hidden,
            hidden_dim=predictor_hidden,
            output_dim=predictor_hidden,
            dim_head=predictor_dim_head,
            dropout=predictor_dropout,
            adaln_rank=predictor_adaln_rank,
        ).to(device)

        self.n_patches = int(n_patches)
        self.emb_dim = int(predictor_hidden)
        self.patch_dim_raw = int(patch_dim)  # 768
        self.action_frameskip = int(action_frameskip)
        self.action_dim = int(action_dim)
        self.device_str = device

    # ------------------------------------------------------------------
    # Training / planning API
    # ------------------------------------------------------------------

    def _patches_from_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) -> (B, T, N, patch_dim_raw=768), float32."""
        return self.backbone.encode(pixels)

    def _project_patches(self, patches_raw: torch.Tensor) -> torch.Tensor:
        """(B, T, N, 768) -> (B, T, N, hidden=256)."""
        return self.patch_proj(patches_raw)

    def _fuse_proprio(
        self,
        patch_emb: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        """Broadcast-add proprio embedding to every patch of each frame.

        patch_emb: (B, T, N, hidden)
        proprio:   (B, T, D_prop)
        returns:   (B, T, N, hidden)
        """
        if self.proprio_encoder is None:
            return patch_emb
        prop_emb = self.proprio_encoder(proprio)  # (B, T, hidden)
        return patch_emb + prop_emb.unsqueeze(2)  # broadcast over N

    def encode(self, info: dict) -> dict:
        """Encode pixels (+optional proprio) into patch latent.

        info must contain "pixels": (B, T, 3, H, W).
        Optionally "proprio": (B, T, D_prop).
        Optionally "action": (B, T, action_dim*frameskip).

        Writes info["emb"]   = (B, T, N=49, hidden=256) and
               info["act_emb"] = (B, T, hidden=256) if action present.
        """
        pixels = info["pixels"]
        patches_raw = self._patches_from_pixels(pixels)  # (B, T, N, 768)
        emb = self._project_patches(patches_raw)         # (B, T, N, hidden)
        if "proprio" in info:
            emb = self._fuse_proprio(emb, info["proprio"].float())
        info["emb"] = emb

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict(
        self, emb: torch.Tensor, act_emb: torch.Tensor
    ) -> torch.Tensor:
        """One-shot prediction of next-step patch tensors.

        emb:     (B, T, N, hidden)
        act_emb: (B, T, hidden)
        returns: (B, T, N, hidden)
        """
        return self.predictor(emb, act_emb)

    # ------------------------------------------------------------------
    # Inference rollout (for LowLevelPlanner MPPI)
    # ------------------------------------------------------------------

    def rollout(
        self,
        info: dict,
        action_sequence: torch.Tensor,
        history_size: int = 3,
    ) -> dict:
        """Autoregressive latent rollout matching JEPAv4/LeWM API.

        Accepts EITHER pre-encoded ``"emb"`` OR raw ``"pixels"`` in
        ``info``. Pre-encoded is strictly preferred at plan time — it
        lets the caller (e.g. ``EncoderHistory``) amortize the DINOv3
        forward across MPPI iterations instead of paying it every time.

        pre-encoded ``emb``: (B, S, T_hist, N, hidden) — straight into rollout.
        pixels:              (B, S, T_hist, 3, H, W) — encoded once internally.
        action_sequence:     (B, S, T, action_dim*frameskip)
                             T_hist and T include both hist and future.
        returns info with "predicted_emb": (B, S, T_total, N, hidden)
        """
        if "emb" in info:
            emb_init = info["emb"]  # (B, S, T_hist, N, hidden)
            H = emb_init.size(2)
        else:
            assert "pixels" in info, (
                "rollout() needs either info['emb'] (pre-encoded) or "
                "info['pixels'] (raw pixels)"
            )
            H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        if "emb" in info:
            # Pre-encoded: emb_init is already (B, S, T_hist, N, hidden).
            emb = info["emb"] = emb_init
        else:
            # Encode the initial history window for a single sample-slice.
            _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            _init = self.encode(_init)
            # _init["emb"]: (B, H, N, hidden)
            emb = info["emb"] = (
                _init["emb"].unsqueeze(1).expand(B, S, -1, -1, -1)
            )  # (B, S, H, N, hidden)
            _init = {k: _detach_clone(v) for k, v in _init.items()}

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()   # (B*S, H, N, hidden)
        act = rearrange(act_0, "b s ... -> (b s) ...")         # (B*S, H, act_dim)
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)           # (B*S, T_cur, hidden)
            emb_trunc = emb[:, -HS:]                     # (B*S, HS, N, hidden)
            act_trunc = act_emb[:, -HS:]                 # (B*S, HS, hidden)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
            # pred_emb: (B*S, 1, N, hidden)
            emb = torch.cat([emb, pred_emb], dim=1)      # (B*S, T_cur+1, N, hidden)

            next_act = act_future[:, t : t + 1, :]
            act = torch.cat([act, next_act], dim=1)

        # Final step.
        act_emb = self.action_encoder(act)
        emb_trunc = emb[:, -HS:]
        act_trunc = act_emb[:, -HS:]
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)

        pred_rollout = rearrange(
            emb, "(b s) ... -> b s ...", b=B, s=S
        )  # (B, S, T_total, N, hidden)
        info["predicted_emb"] = pred_rollout
        return info


class JEPAv5PlannerBackbone(nn.Module):
    """Thin adapter exposing the (projected, proprio-free) patch
    embedding path as a ``.encode(pixels) -> (B, T, N, hidden)`` method.

    The planner's existing API expects an object with ``encode(pixels)``
    and an ``output_dim`` attribute. For v5 the "planner backbone" is
    DINOv3 patch extraction composed with the trainable patch projector
    (no proprio, no predictor). Exposing this as a separate object lets
    ``m4c_hierarchical_plan.py`` and ``LowLevelPlanner`` stay unchanged.
    """

    def __init__(self, jepa: "JEPAv5") -> None:
        super().__init__()
        self.jepa = jepa
        self.output_dim = jepa.emb_dim  # hidden dim (256)
        self.n_patches = jepa.n_patches

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        raw = self.jepa.backbone.encode(pixels)   # (B, T, N, 768)
        return self.jepa.patch_proj(raw)          # (B, T, N, hidden)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.encode(pixels)


def load_jepa_v5_from_checkpoint(
    ckpt_path: str,
    *,
    device: str = "cuda",
) -> JEPAv5:
    """Load a JEPAv5 checkpoint saved by ``scripts/m4f_train_jepa_v5.py``.

    Schema (identical shape to v4 loader):
        {
            "config": dict of __init__ kwargs,
            "state_dict": {"patch_proj.*", "proprio_encoder.*",
                           "action_encoder.*", "predictor.*"},
            ...
        }
    The frozen DINOv3 backbone is reloaded from HF.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    config["device"] = device
    model = JEPAv5(**config)

    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    real_missing = [k for k in missing if not k.startswith("backbone.dinov3.")]
    if real_missing:
        raise RuntimeError(
            f"JEPAv5 load: real missing keys: {real_missing[:5]} "
            f"(total {len(real_missing)})"
        )
    unexpected = [k for k in unexpected if not k.startswith("backbone.dinov3.")]
    if unexpected:
        raise RuntimeError(
            f"JEPAv5 load: unexpected keys: {unexpected[:5]} "
            f"(total {len(unexpected)})"
        )
    model = model.to(device).eval()
    return model
