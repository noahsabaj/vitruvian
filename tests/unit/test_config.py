# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Config loader — dotted overrides + type coercion."""

from __future__ import annotations

from pathlib import Path

import pytest

from vitruvian.utils import load_config


def test_load_yaml(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("trainer:\n  lr: 1.0e-4\n  batch_size: 32\n")
    cfg = load_config(p)
    assert cfg["trainer"]["lr"] == 1e-4
    assert cfg["trainer"]["batch_size"] == 32


def test_override_coerces_numeric(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("trainer:\n  lr: 1.0e-4\n")
    cfg = load_config(p, overrides=["trainer.lr=3e-5"])
    assert cfg["trainer"]["lr"] == 3e-5
    assert isinstance(cfg["trainer"]["lr"], float)


def test_override_coerces_bool_and_none(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("trainer:\n  bf16: true\n  warmup: 500\n")
    cfg = load_config(p, overrides=["trainer.bf16=false", "trainer.warmup=null"])
    assert cfg["trainer"]["bf16"] is False
    assert cfg["trainer"]["warmup"] is None


def test_override_creates_new_path(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("{}\n")
    cfg = load_config(p, overrides=["a.b.c=hello"])
    assert cfg["a"]["b"]["c"] == "hello"


def test_bad_override_raises(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("{}\n")
    with pytest.raises(ValueError):
        load_config(p, overrides=["not-a-kv-pair"])


def test_override_into_scalar_raises(tmp_path: Path) -> None:
    """Descending through a scalar node is almost always a typo — raise a clear
    error instead of silently clobbering the scalar with a new mapping."""
    p = tmp_path / "cfg.yaml"
    p.write_text("trainer: 5\n")
    with pytest.raises(ValueError, match="not a mapping"):
        load_config(p, overrides=["trainer.lr=1e-4"])
