# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Registry — unknown names, checkpoint round-trip with migration."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vitruvian.models import JEPA, build_jepa, load_jepa
from vitruvian.models.registry import BACKBONES


def test_registry_excludes_lewm_v3() -> None:
    """`lewm-v3` was removed from BACKBONES in M4.9.1 — it's a stub
    (``LeWMBackbone`` takes a pre-constructed JEPA, can't be built from
    YAML kwargs). Legacy v3 checkpoints load via
    ``load_lewm_jepa_from_checkpoint`` directly.
    """
    assert "lewm-v3" not in BACKBONES
    assert set(BACKBONES.keys()) == {"dinov3-cls", "dinov3-patch"}


def test_unknown_backbone_raises(registry_with_fakes) -> None:
    cfg = {
        "backbone": {"name": "does-not-exist"},
        "predictor": {"name": "ar", "kwargs": {"num_frames": 3, "depth": 1,
                                                "heads": 2, "mlp_dim": 32,
                                                "input_dim": 32, "hidden_dim": 32,
                                                "output_dim": 32, "dim_head": 16,
                                                "dropout": 0.0}},
        "action_encoder": {"kwargs": {"action_dim": 29}},
    }
    with pytest.raises(KeyError):
        build_jepa(cfg)


def test_unknown_predictor_raises(registry_with_fakes) -> None:
    cfg = {
        "backbone": {"name": "dinov3-cls", "kwargs": {}},
        "predictor": {"name": "does-not-exist"},
        "action_encoder": {"kwargs": {"action_dim": 29}},
    }
    with pytest.raises(KeyError):
        build_jepa(cfg)


def test_v5_legacy_ckpt_migration(
    tmp_path: Path, registry_with_fakes
) -> None:
    """Save a legacy v5 config + patch_proj.* keys, confirm load_jepa
    migrates to the unified schema."""
    cfg_v5 = {
        "dinov3_model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "spatial_stride": 2,
        "proprio_dim": 103,
        "proprio_hidden": 64,
        "action_dim": 29,
        "action_frameskip": 1,
        "action_smoothed_dim": 10,
        "predictor_num_frames": 3,
        "predictor_depth": 2,
        "predictor_heads": 4,
        "predictor_mlp_dim": 256,
        "predictor_hidden": 128,
        "predictor_dim_head": 32,
        "predictor_dropout": 0.0,
        "predictor_adaln_rank": 64,
        "backbone_lazy": True,
        "device": "cpu",
    }

    # Build a unified model from the migrated v5 config to mint an
    # appropriate state_dict, then re-key ``patch_projector.*`` →
    # ``patch_proj.*`` to simulate a legacy checkpoint.
    from vitruvian.models.registry import _migrate_v5_config

    unified_cfg = _migrate_v5_config(cfg_v5)
    unified_cfg["backbone"]["kwargs"]["device"] = "cpu"
    model = build_jepa(unified_cfg)
    sd = model.state_dict()

    legacy_sd = {}
    for k, v in sd.items():
        if k.startswith("patch_projector."):
            legacy_sd["patch_proj." + k[len("patch_projector."):]] = v
        else:
            legacy_sd[k] = v

    ckpt_path = tmp_path / "legacy_v5.pt"
    torch.save({"config": cfg_v5, "state_dict": legacy_sd}, ckpt_path)

    # Legacy checkpoint (no arch_version) → load_jepa warns a retrain is
    # needed, but still loads for schema-migration verification.
    with pytest.warns(UserWarning, match="arch_version"):
        loaded = load_jepa(ckpt_path, device="cpu")
    new_sd = loaded.state_dict()
    for k, v in sd.items():
        if k.startswith("backbone."):
            continue
        assert k in new_sd, f"missing {k}"
        assert torch.equal(new_sd[k], v), f"{k} mismatch"
