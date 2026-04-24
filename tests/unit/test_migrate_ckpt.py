# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Tests for scripts/migrate_ckpt.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

# Import the script as a module.
_MIGRATE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "migrate_ckpt.py"
_spec = importlib.util.spec_from_file_location("migrate_ckpt", _MIGRATE_PATH)
migrate_ckpt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migrate_ckpt)


def _make_legacy_v5_ckpt(tmp_path: Path, registry_with_fakes) -> Path:
    """Build a v5-shaped legacy checkpoint on disk + return its path."""
    from vitruvian.models import build_jepa
    from vitruvian.models.registry import _migrate_v5_config

    legacy_cfg = {
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
    unified = _migrate_v5_config(legacy_cfg)
    unified["backbone"]["kwargs"]["device"] = "cpu"
    m = build_jepa(unified)
    sd = m.state_dict()
    # Rewrite patch_projector.* → patch_proj.* so we simulate legacy keys.
    legacy_sd = {}
    for k, v in sd.items():
        if k.startswith("patch_projector."):
            legacy_sd["patch_proj." + k[len("patch_projector."):]] = v
        else:
            legacy_sd[k] = v

    ckpt_path = tmp_path / "legacy_v5.pt"
    torch.save({"config": legacy_cfg, "state_dict": legacy_sd}, ckpt_path)
    return ckpt_path


def test_detects_legacy_v5(tmp_path: Path, registry_with_fakes) -> None:
    path = _make_legacy_v5_ckpt(tmp_path, registry_with_fakes)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert not migrate_ckpt.is_unified_schema(ckpt["config"])


def test_migrate_in_place_with_backup(
    tmp_path: Path, registry_with_fakes
) -> None:
    src = _make_legacy_v5_ckpt(tmp_path, registry_with_fakes)
    assert migrate_ckpt.migrate_ckpt(src) is True

    # Backup should exist.
    bak = src.with_suffix(src.suffix + ".bak")
    assert bak.exists()

    # Source is now the unified schema.
    migrated = torch.load(src, map_location="cpu", weights_only=False)
    assert migrate_ckpt.is_unified_schema(migrated["config"])
    assert migrated["migrated_from"] == "v5"


def test_migrate_idempotent(tmp_path: Path, registry_with_fakes) -> None:
    src = _make_legacy_v5_ckpt(tmp_path, registry_with_fakes)
    migrate_ckpt.migrate_ckpt(src)
    # Second call: already unified — should return False and not change the file.
    mtime_before = src.stat().st_mtime
    import time
    time.sleep(0.01)
    assert migrate_ckpt.migrate_ckpt(src, make_backup=False) is False
    assert src.stat().st_mtime == mtime_before


def test_migrate_copy_mode(tmp_path: Path, registry_with_fakes) -> None:
    src = _make_legacy_v5_ckpt(tmp_path, registry_with_fakes)
    dst = src.with_suffix(src.suffix + ".migrated.pt")
    migrate_ckpt.migrate_ckpt(src, dst=dst, make_backup=False)
    assert dst.exists()
    # No .bak written in copy mode.
    assert not src.with_suffix(src.suffix + ".bak").exists()
    # Source unchanged.
    src_ckpt = torch.load(src, map_location="cpu", weights_only=False)
    assert not migrate_ckpt.is_unified_schema(src_ckpt["config"])


def test_migrated_file_loads_via_load_jepa(
    tmp_path: Path, registry_with_fakes
) -> None:
    from vitruvian.models import load_jepa

    src = _make_legacy_v5_ckpt(tmp_path, registry_with_fakes)
    migrate_ckpt.migrate_ckpt(src, make_backup=False)
    loaded = load_jepa(src, device="cpu")
    assert loaded.emb_dim == 128
    assert loaded.n_patches == 49


def test_dry_run_writes_nothing(tmp_path: Path, registry_with_fakes) -> None:
    src = _make_legacy_v5_ckpt(tmp_path, registry_with_fakes)
    mtime_before = src.stat().st_mtime
    import time
    time.sleep(0.01)
    migrate_ckpt.migrate_ckpt(src, dry_run=True)
    assert src.stat().st_mtime == mtime_before
    assert not src.with_suffix(src.suffix + ".bak").exists()


def test_non_ckpt_file_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bogus.pt"
    torch.save({"nonsense": 1}, bad)
    with pytest.raises(RuntimeError, match="missing 'config'"):
        migrate_ckpt.migrate_ckpt(bad)
