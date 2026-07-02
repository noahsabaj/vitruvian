# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Shared fixtures — tiny synthetic HDF5 + mocked backbones.

Keeps the whole suite CPU-only and under 60s by avoiding any HuggingFace
weight download. The synthetic HDF5 mirrors the full schema of the G1
expert dataset (pixels, proprio, action, state, ep_offset, ep_len,
commands) so dataset/merge code sees realistic layouts.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

# Put src/ on sys.path first so tests exercise the local package.
import sys

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


@pytest.fixture(scope="session")
def synthetic_h5(tmp_path_factory) -> Path:
    """Build a tiny 2-episode × 25-step G1 expert-shaped HDF5."""
    path = tmp_path_factory.mktemp("data") / "synthetic.h5"
    n_ep = 2
    ep_len = 25
    total = n_ep * ep_len
    rng = np.random.default_rng(0)

    with h5py.File(path, "w") as f:
        f.create_dataset(
            "pixels",
            data=rng.integers(0, 255, (total, 224, 224, 3), dtype=np.uint8),
        )
        f.create_dataset(
            "proprio",
            data=rng.standard_normal((total, 103), dtype=np.float32),
        )
        f.create_dataset(
            "action",
            data=rng.standard_normal((total, 29), dtype=np.float32),
        )
        f.create_dataset(
            "state",
            data=rng.standard_normal((total, 60), dtype=np.float32),
        )
        f.create_dataset(
            "ep_offset", data=np.asarray([0, ep_len], dtype=np.int64)
        )
        f.create_dataset(
            "ep_len", data=np.asarray([ep_len, ep_len], dtype=np.int64)
        )
        f.create_dataset(
            "commands",
            data=np.tile(
                np.asarray([0.5, 0.0, 0.0], dtype=np.float32), (n_ep, 1)
            ),
        )
    return path


class FakeClsBackbone(nn.Module):
    """Deterministic stand-in for DINOv3 CLS backbone (no HF download)."""

    output_dim = 768

    def __init__(self, **_) -> None:
        super().__init__()
        # Seed so outputs are reproducible across test invocations.
        self._rng = torch.Generator().manual_seed(0)
        self._proj = nn.Linear(3 * 224 * 224, self.output_dim, bias=False)
        with torch.no_grad():
            self._proj.weight.normal_(
                generator=self._rng, mean=0.0, std=1e-4
            )
        for p in self._proj.parameters():
            p.requires_grad_(False)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        B, T = pixels.size(0), pixels.size(1)
        x = pixels.float()
        if x.max() > 1.5:
            x = x / 255.0
        flat = x.reshape(B * T, -1)
        return self._proj(flat).reshape(B, T, self.output_dim)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.encode(pixels)


class FakePatchBackbone(nn.Module):
    """Deterministic stand-in for DINOv3 patch backbone.

    Produces per-patch-*distinct* tokens — each of the ``n_patches``
    tokens is projected from its own pooled spatial cell — so tests
    actually exercise spatial structure (per-patch positional
    embeddings, spatial attention, the patch-mean cosine) instead of
    ``n_patches`` identical copies of one global vector.
    """

    output_dim = 768
    n_patches = 49

    def __init__(self, **_) -> None:
        super().__init__()
        self._rng = torch.Generator().manual_seed(1)
        self._side = int(self.n_patches**0.5)  # 7×7 = 49 cells
        self._patch_proj = nn.Linear(3, self.output_dim, bias=False)
        with torch.no_grad():
            self._patch_proj.weight.normal_(
                generator=self._rng, mean=0.0, std=1e-2
            )
        for p in self._patch_proj.parameters():
            p.requires_grad_(False)

    def load_eagerly(self) -> None:  # no-op, keeps duck-typing happy
        return

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        B, T = pixels.size(0), pixels.size(1)
        x = pixels.float()
        if x.max() > 1.5:
            x = x / 255.0
        # Pool to an N-cell grid; project each cell's RGB independently so
        # the N patch tokens carry distinct, input-dependent signal.
        x = torch.nn.functional.adaptive_avg_pool2d(
            x.reshape(B * T, 3, 224, 224), (self._side, self._side)
        )  # (B*T, 3, side, side)
        x = x.permute(0, 2, 3, 1).reshape(B * T, self.n_patches, 3)
        out = self._patch_proj(x)  # (B*T, N, 768) — distinct per patch
        return out.reshape(B, T, self.n_patches, self.output_dim)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.encode(pixels)


@pytest.fixture
def fake_cls_backbone() -> FakeClsBackbone:
    return FakeClsBackbone()


@pytest.fixture
def fake_patch_backbone() -> FakePatchBackbone:
    return FakePatchBackbone()


@pytest.fixture
def registry_with_fakes(monkeypatch, fake_cls_backbone, fake_patch_backbone):
    """Swap the registry's backbone entries for deterministic fakes."""
    from vitruvian.models import registry as reg

    orig = dict(reg.BACKBONES)
    monkeypatch.setitem(reg.BACKBONES, "dinov3-cls", FakeClsBackbone)
    monkeypatch.setitem(reg.BACKBONES, "dinov3-patch", FakePatchBackbone)
    yield reg
    reg.BACKBONES.clear()
    reg.BACKBONES.update(orig)
