# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""JEPA registry — build/load entrypoints.

:func:`build_jepa` constructs a :class:`JEPA` from a config dict that
names the backbone (``dinov3-cls`` / ``dinov3-patch`` / ``lewm-v3``),
the predictor (``ar`` / ``patch-ar``), and the action/proprio/projector
kwargs.

:func:`load_jepa` is the **single** checkpoint entry point that
subsumes the three historical loaders (``load_jepa_v4_from_checkpoint``,
``load_jepa_v5_from_checkpoint``, ``load_lewm_jepa_from_checkpoint``).
It dispatches on the ``config`` dict stored in the checkpoint — new
checkpoints save the unified schema; legacy v4/v5 checkpoints are
transparently migrated in memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vitruvian.lewm_compat import MLP as LeWMMLP
from vitruvian.lewm_compat import Embedder as LeWMEmbedder
from vitruvian.models.backbones import (
    Backbone,
    DEFAULT_DINOV3_ID,
    DINOv3ClsBackbone,
    DINOv3PatchBackbone,
)
from vitruvian.models.jepa import JEPA
from vitruvian.models.predictors import ARPredictor, PatchARPredictor

BACKBONES: dict[str, type[Backbone]] = {
    "dinov3-cls": DINOv3ClsBackbone,
    "dinov3-patch": DINOv3PatchBackbone,
    # Note: ``LeWMBackbone`` is intentionally NOT in this registry. It
    # wraps a pre-constructed LeWM JEPA object and cannot be built from
    # YAML kwargs — load legacy v3 checkpoints through
    # ``vitruvian.models.load_lewm_jepa_from_checkpoint`` directly.
}
PREDICTORS: dict[str, type[nn.Module]] = {
    "ar": ARPredictor,
    "patch-ar": PatchARPredictor,
}


# --------------------------------------------------------------------------
# Build from config dict
# --------------------------------------------------------------------------


def _build_backbone(cfg: dict[str, Any]) -> Backbone:
    name = cfg["name"]
    if name not in BACKBONES:
        raise KeyError(f"unknown backbone {name!r}; known: {sorted(BACKBONES)}")
    kwargs = dict(cfg.get("kwargs", {}))
    return BACKBONES[name](**kwargs)


def _build_predictor(
    cfg: dict[str, Any], *, n_patches: int | None
) -> nn.Module:
    name = cfg["name"]
    if name not in PREDICTORS:
        raise KeyError(f"unknown predictor {name!r}; known: {sorted(PREDICTORS)}")
    kwargs = dict(cfg.get("kwargs", {}))
    if name == "patch-ar" and "num_patches" not in kwargs:
        if n_patches is None:
            raise ValueError(
                "patch-ar predictor needs num_patches; either set it in "
                "cfg.predictor.kwargs or use a patch-shaped backbone"
            )
        kwargs["num_patches"] = int(n_patches)
    return PREDICTORS[name](**kwargs)


def _build_action_encoder(cfg: dict[str, Any], *, emb_dim: int) -> nn.Module:
    kwargs = dict(cfg.get("kwargs", {}))
    action_dim = int(kwargs.pop("action_dim"))
    frameskip = int(kwargs.pop("frameskip", 1))
    smoothed_dim = int(kwargs.pop("smoothed_dim", 10))
    kwargs.setdefault("emb_dim", emb_dim)
    return LeWMEmbedder(
        input_dim=action_dim * frameskip,
        smoothed_dim=smoothed_dim,
        **kwargs,
    )


def _build_proprio_encoder(
    cfg: dict[str, Any] | None, *, emb_dim: int
) -> nn.Module | None:
    if cfg is None:
        return None
    kwargs = dict(cfg.get("kwargs", {}))
    in_dim = int(kwargs.pop("in_dim"))
    hidden = int(kwargs.pop("hidden", 256))
    out_dim = int(kwargs.pop("out_dim", emb_dim))
    norm_fn = kwargs.pop("norm_fn", None)
    return LeWMMLP(
        input_dim=in_dim,
        output_dim=out_dim,
        hidden_dim=hidden,
        norm_fn=norm_fn,
    )


def _build_patch_projector(
    cfg: dict[str, Any] | None,
) -> nn.Module | None:
    if cfg is None:
        return None
    kwargs = dict(cfg.get("kwargs", {}))
    return nn.Linear(int(kwargs["in_dim"]), int(kwargs["out_dim"]))


def build_jepa(cfg: dict[str, Any]) -> JEPA:
    """Build a :class:`JEPA` from a config dict.

    Expected schema::

        {
          "backbone": {"name": str, "kwargs": {...}},
          "predictor": {"name": str, "kwargs": {...}},
          "action_encoder": {"kwargs": {"action_dim": int, ...}},
          "proprio_encoder": {"kwargs": {"in_dim": int, ...}} | None,
          "patch_projector": {"kwargs": {"in_dim": int, "out_dim": int}}
                             | None,
        }
    """
    backbone = _build_backbone(cfg["backbone"])
    patch_projector = _build_patch_projector(cfg.get("patch_projector"))
    predictor = _build_predictor(
        cfg["predictor"], n_patches=getattr(backbone, "n_patches", None)
    )

    # Predictor's hidden_dim drives action/proprio embedding widths.
    emb_dim = int(getattr(predictor, "hidden_dim", backbone.output_dim))

    action_encoder = _build_action_encoder(cfg["action_encoder"], emb_dim=emb_dim)
    proprio_encoder = _build_proprio_encoder(
        cfg.get("proprio_encoder"), emb_dim=emb_dim
    )
    return JEPA(
        backbone=backbone,
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
        patch_projector=patch_projector,
    )


# --------------------------------------------------------------------------
# Load + checkpoint migration
# --------------------------------------------------------------------------


def _migrate_v4_config(cfg_v4: dict[str, Any]) -> dict[str, Any]:
    """Translate a v4 ``__init__``-kwargs dict into the unified schema."""
    return {
        "backbone": {
            "name": "dinov3-cls",
            "kwargs": {
                "model_id": cfg_v4.get(
                    "dinov3_model_id", DEFAULT_DINOV3_ID
                ),
                "dtype": cfg_v4.get("dinov3_dtype", torch.float16),
                "device": cfg_v4.get("device", "cuda"),
                "freeze": True,
            },
        },
        "predictor": {
            "name": "ar",
            "kwargs": {
                "num_frames": int(cfg_v4.get("predictor_num_frames", 3)),
                "depth": int(cfg_v4.get("predictor_depth", 6)),
                "heads": int(cfg_v4.get("predictor_heads", 16)),
                "mlp_dim": int(cfg_v4.get("predictor_mlp_dim", 3072)),
                "input_dim": 768,
                "hidden_dim": 768,
                "output_dim": 768,
                "dim_head": int(cfg_v4.get("predictor_dim_head", 64)),
                "dropout": float(cfg_v4.get("predictor_dropout", 0.1)),
                "emb_dropout": 0.0,
            },
        },
        "action_encoder": {
            "kwargs": {
                "action_dim": int(cfg_v4.get("action_dim", 29)),
                "frameskip": int(cfg_v4.get("action_frameskip", 1)),
                "smoothed_dim": int(cfg_v4.get("action_smoothed_dim", 10)),
                "emb_dim": 768,
            },
        },
        "proprio_encoder": (
            {
                "kwargs": {
                    "in_dim": int(cfg_v4["proprio_dim"]),
                    "hidden": int(cfg_v4.get("proprio_hidden", 256)),
                    "out_dim": 768,
                    "norm_fn": None,
                }
            }
            if (cfg_v4.get("proprio_dim") or 0) > 0
            else None
        ),
        "patch_projector": None,
    }


def _migrate_v5_config(cfg_v5: dict[str, Any]) -> dict[str, Any]:
    """Translate a v5 ``__init__``-kwargs dict into the unified schema."""
    hidden = int(cfg_v5.get("predictor_hidden", 256))
    return {
        "backbone": {
            "name": "dinov3-patch",
            "kwargs": {
                "model_id": cfg_v5.get(
                    "dinov3_model_id", DEFAULT_DINOV3_ID
                ),
                "dtype": cfg_v5.get("dinov3_dtype", torch.float16),
                "device": cfg_v5.get("device", "cuda"),
                "freeze": True,
                "spatial_stride": int(cfg_v5.get("spatial_stride", 2)),
                "lazy": bool(cfg_v5.get("backbone_lazy", False)),
            },
        },
        "predictor": {
            "name": "patch-ar",
            "kwargs": {
                "num_frames": int(cfg_v5.get("predictor_num_frames", 3)),
                "depth": int(cfg_v5.get("predictor_depth", 6)),
                "heads": int(cfg_v5.get("predictor_heads", 8)),
                "mlp_dim": int(cfg_v5.get("predictor_mlp_dim", 1024)),
                "input_dim": hidden,
                "hidden_dim": hidden,
                "output_dim": hidden,
                "dim_head": int(cfg_v5.get("predictor_dim_head", 32)),
                "dropout": float(cfg_v5.get("predictor_dropout", 0.1)),
                "adaln_rank": int(cfg_v5.get("predictor_adaln_rank", 128)),
                # num_patches is inferred from the backbone at build time.
            },
        },
        "action_encoder": {
            "kwargs": {
                "action_dim": int(cfg_v5.get("action_dim", 29)),
                "frameskip": int(cfg_v5.get("action_frameskip", 1)),
                "smoothed_dim": int(cfg_v5.get("action_smoothed_dim", 10)),
                "emb_dim": hidden,
            },
        },
        "proprio_encoder": (
            {
                "kwargs": {
                    "in_dim": int(cfg_v5["proprio_dim"]),
                    "hidden": int(cfg_v5.get("proprio_hidden", 256)),
                    "out_dim": hidden,
                    "norm_fn": None,
                }
            }
            if (cfg_v5.get("proprio_dim") or 0) > 0
            else None
        ),
        "patch_projector": {
            "kwargs": {"in_dim": 768, "out_dim": hidden},
        },
    }


def _detect_legacy_schema(cfg: dict[str, Any]) -> str | None:
    """Return 'v4' / 'v5' / None based on tell-tale legacy keys."""
    if "spatial_stride" in cfg or "predictor_hidden" in cfg:
        return "v5"
    if "dinov3_model_id" in cfg or "predictor_mlp_dim" in cfg:
        return "v4"
    return None


def _normalize_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip compile/DDP wrappers and map v5 ``patch_proj.*`` keys to
    the unified ``patch_projector.*`` path."""
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        kk = k
        kk = kk.replace("_orig_mod.", "")
        if kk.startswith("patch_proj."):
            kk = "patch_projector." + kk[len("patch_proj.") :]
        out[kk] = v
    return out


def load_jepa(
    ckpt_path: str | Path,
    *,
    device: str = "cuda",
) -> JEPA:
    """Load a :class:`JEPA` from a unified-schema or legacy v4/v5 checkpoint.

    Supports:

    * **Unified** checkpoints (``config`` is already in the new schema).
    * **Legacy v4** checkpoints (config carries ``dinov3_model_id`` +
      ``predictor_mlp_dim`` but no ``spatial_stride``) — migrated in
      memory to the unified schema.
    * **Legacy v5** checkpoints (config carries ``spatial_stride`` or
      ``predictor_hidden``) — migrated in memory; ``patch_proj.*``
      state_dict keys re-homed to ``patch_projector.*``.

    **Does NOT handle v3 LeWM Lightning checkpoints.** Those have a
    different schema (encoder.*, predictor.*, action_encoder.* with
    their own hyperparameters) and a different composition class.
    Load them via :func:`vitruvian.models.load_lewm_jepa_from_checkpoint`.

    The frozen backbone is reloaded from HuggingFace; its weights are
    not persisted in the checkpoint. ``backbone.*`` keys missing from
    the checkpoint are expected and filtered from the strictness check.
    """
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "config" not in ckpt:
        raise RuntimeError(
            f"checkpoint at {ckpt_path} has no 'config' — cannot dispatch"
        )

    cfg = dict(ckpt["config"])
    legacy = _detect_legacy_schema(cfg)
    if legacy == "v4":
        cfg = _migrate_v4_config(cfg)
    elif legacy == "v5":
        cfg = _migrate_v5_config(cfg)
    else:
        # Unified schema — ensure device override.
        bb_kwargs = cfg.setdefault("backbone", {}).setdefault("kwargs", {})
        bb_kwargs["device"] = device

    # Backbone-level device override for legacy migration too.
    cfg["backbone"]["kwargs"]["device"] = device

    model = build_jepa(cfg)
    state = _normalize_state_dict(ckpt["state_dict"])
    missing, unexpected = model.load_state_dict(state, strict=False)

    real_missing = [
        k
        for k in missing
        if not (
            k.startswith("backbone.dinov3.")
            or k.startswith("backbone.jepa.")
        )
    ]
    if real_missing:
        raise RuntimeError(
            f"load_jepa: missing keys: {real_missing[:5]} "
            f"(total {len(real_missing)})"
        )
    real_unexpected = [
        k
        for k in unexpected
        if not (
            k.startswith("backbone.dinov3.")
            or k.startswith("backbone.jepa.")
        )
    ]
    if real_unexpected:
        raise RuntimeError(
            f"load_jepa: unexpected keys: {real_unexpected[:5]} "
            f"(total {len(real_unexpected)})"
        )

    return model.to(device).eval()


__all__ = [
    "BACKBONES",
    "PREDICTORS",
    "build_jepa",
    "load_jepa",
]
