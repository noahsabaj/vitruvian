# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""M4.5 JEPAv4 — frozen DINOv3 + trainable proprio MLP + ARPredictor.

Composes ``DINOv3Backbone`` (frozen, image-only) with LeWM's
``ARPredictor`` and ``Embedder`` (action encoder) plus a new trainable
``MLP`` (103 → 256 → 768) for proprioception. Exposes the same
``encode`` / ``predict`` / ``rollout`` API as ``external/le-wm/jepa.py``
``JEPA`` so ``LowLevelPlanner`` in ``src/vitruvian/hwm/planners.py``
needs no changes.

Design choice: the proprio branch is a separate trainable MLP whose
output is **summed** with the frozen DINOv3 CLS before entering the
predictor — this is the Terver et al. (arXiv:2512.24497) feature-
conditioning recipe and matches the pattern in LeWM's jepa.encode.

Projector and pred_proj are ``nn.Identity`` — the frozen DINOv3 768-D
output already is the JEPA latent; we don't re-embed it.

Training objective lives outside this module (see scripts/m4e_train_jepa_v4.py):
standard 1-step teacher-forcing MSE + k-step rollout MSE. SIGReg is
dropped — encoder is frozen, no collapse to prevent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange

# Pull ARPredictor, Embedder, MLP from LeWM without needing stable_pretraining.
_ROOT = Path(__file__).resolve().parents[3]
_LEWM_SRC = _ROOT / "external" / "le-wm"
if str(_LEWM_SRC) not in sys.path:
    sys.path.insert(0, str(_LEWM_SRC))

import module as lewm_module  # type: ignore[import-not-found]  # noqa: E402

from vitruvian.hwm.backbone_dinov3 import DINOv3Backbone  # noqa: E402


def _detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


class JEPAv4(nn.Module):
    """DINOv3-backed world model with LeWM-compatible API.

    .encode(info)  — pixels (+ optional proprio) → emb (B, T, 768).
                     If action present in info, also writes act_emb.
    .predict(emb, act_emb) — (B, T, 768) → (B, T, 768).
    .rollout(info, action_seq, history_size) — for LowLevelPlanner MPPI.
    """

    def __init__(
        self,
        *,
        dinov3_model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        proprio_dim: int | None = 103,
        proprio_hidden: int = 256,
        action_dim: int = 29,
        action_frameskip: int = 1,
        action_smoothed_dim: int = 10,
        predictor_num_frames: int = 3,
        predictor_depth: int = 6,
        predictor_heads: int = 16,
        predictor_mlp_dim: int = 3072,
        predictor_dim_head: int = 64,
        predictor_dropout: float = 0.1,
        device: str = "cuda",
        dinov3_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()

        # Visual (frozen).
        self.backbone = DINOv3Backbone(
            model_id=dinov3_model_id,
            device=device,
            dtype=dinov3_dtype,
            freeze=True,
        )
        emb_dim = self.backbone.output_dim  # 768

        # Proprio (trainable).
        if proprio_dim is not None and proprio_dim > 0:
            self.proprio_encoder = lewm_module.MLP(
                input_dim=int(proprio_dim),
                output_dim=emb_dim,
                hidden_dim=proprio_hidden,
                norm_fn=None,
            ).to(device)
        else:
            self.proprio_encoder = None

        # Action encoder (trainable).
        self.action_encoder = lewm_module.Embedder(
            input_dim=int(action_dim) * int(action_frameskip),
            smoothed_dim=int(action_smoothed_dim),
            emb_dim=emb_dim,
        ).to(device)

        # Predictor (trainable).
        self.predictor = lewm_module.ARPredictor(
            num_frames=int(predictor_num_frames),
            depth=int(predictor_depth),
            heads=int(predictor_heads),
            mlp_dim=int(predictor_mlp_dim),
            input_dim=emb_dim,
            hidden_dim=emb_dim,
            output_dim=emb_dim,
            dim_head=int(predictor_dim_head),
            dropout=float(predictor_dropout),
            emb_dropout=0.0,
        ).to(device)

        # Projector and pred_proj are identity here — DINOv3 output IS the
        # JEPA latent; no re-projection needed.
        self.projector = nn.Identity()
        self.pred_proj = nn.Identity()

        self.emb_dim = emb_dim
        self.action_frameskip = int(action_frameskip)
        self.action_dim = int(action_dim)
        self.device_str = device

    # ------------------------------------------------------------------
    # Training API — mirrors external/le-wm/jepa.py JEPA
    # ------------------------------------------------------------------

    def encode(self, info: dict) -> dict:
        """Encode pixels (+ optional proprio) to fused 768-D embedding.

        info must contain "pixels": (B, T, 3, H, W) uint8 or float.
        Optionally "proprio": (B, T, D_prop).
        Optionally "action": (B, T, action_dim*frameskip).

        Writes info["emb"] = (B, T, 768) and info["act_emb"] if action
        present. Returns info.
        """
        pixels = info["pixels"]
        emb = self.backbone.encode(pixels)  # (B, T, 768), float32
        emb = self.projector(emb)  # identity

        if self.proprio_encoder is not None and "proprio" in info:
            proprio = info["proprio"].float()  # (B, T, D_prop)
            prop_emb = self.proprio_encoder(proprio)  # (B, T, 768)
            emb = emb + prop_emb

        info["emb"] = emb

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """One-shot prediction of next embeddings given context + actions.

        emb: (B, T, 768), act_emb: (B, T, 768). Returns (B, T, 768).
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    # ------------------------------------------------------------------
    # Inference-only API — used by LowLevelPlanner
    # ------------------------------------------------------------------

    def rollout(
        self,
        info: dict,
        action_sequence: torch.Tensor,
        history_size: int = 3,
    ) -> dict:
        """Autoregressive rollout for MPPI candidate scoring.

        Accepts either pre-encoded ``"emb"`` of shape
        ``(B, S, T_hist, D)`` or raw ``"pixels"`` of shape
        ``(B, S, T_hist, 3, H, W)`` in ``info``. Pre-encoded skips the
        DINOv3 forward at plan time (M4.7 win).

        action_sequence: (B, S, T, action_dim*frameskip) — hist + future.

        Mirrors the LeWM JEPA.rollout signature/semantics so
        LowLevelPlanner doesn't care which backbone is behind it.
        """
        if "emb" in info:
            emb_init = info["emb"]
            H = emb_init.size(2)
        else:
            assert "pixels" in info, (
                "rollout() needs either info['emb'] or info['pixels']"
            )
            H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        if "emb" in info:
            emb = info["emb"] = emb_init  # (B, S, H, D)
        else:
            # Encode the initial history window.
            _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            _init = self.encode(_init)
            emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
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


def load_jepa_v4_from_checkpoint(
    ckpt_path: str,
    *,
    device: str = "cuda",
) -> JEPAv4:
    """Load a JEPAv4 checkpoint saved by scripts/m4e_train_jepa_v4.py.

    Checkpoint schema:
        {
            "config": dict of all __init__ kwargs,
            "state_dict": {"proprio_encoder.*", "action_encoder.*", "predictor.*"},
            ...
        }
    The DINOv3 backbone is reloaded from HuggingFace — we don't persist
    its 86M frozen weights in our checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    config["device"] = device
    model = JEPAv4(**config)

    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    # "dinov3.*" keys are expected to be missing (we reload from HF).
    real_missing = [k for k in missing if not k.startswith("backbone.dinov3.")]
    if real_missing:
        raise RuntimeError(
            f"JEPAv4 load: real missing keys: {real_missing[:5]} "
            f"(total {len(real_missing)})"
        )
    # Unexpected should be empty in our checkpoint format.
    unexpected = [k for k in unexpected if not k.startswith("backbone.dinov3.")]
    if unexpected:
        raise RuntimeError(
            f"JEPAv4 load: unexpected keys: {unexpected[:5]} "
            f"(total {len(unexpected)})"
        )

    model = model.to(device).eval()
    return model
