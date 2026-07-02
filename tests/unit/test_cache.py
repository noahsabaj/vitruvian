# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""EmbeddingCache unit tests."""

from __future__ import annotations

from pathlib import Path

import torch

from vitruvian.data import CacheKey, EmbeddingCache


def test_cache_roundtrip_mmap(tmp_path: Path, synthetic_h5: Path) -> None:
    key = CacheKey(h5_path=synthetic_h5, model_id="fake", mode="cls")

    written = torch.arange(50 * 16, dtype=torch.float32).reshape(50, 16)

    def compute():
        return written, {"model_id": "fake", "dim": 16}

    # MISS
    cache = EmbeddingCache.from_precompute(
        cache_dir=tmp_path, key=key, compute_fn=compute
    )
    assert cache.tensor.shape == written.shape
    assert torch.equal(cache.tensor, written)

    # HIT — should not call compute()
    called = {"n": 0}

    def recompute():
        called["n"] += 1
        return written, {"model_id": "fake", "dim": 16}

    cache2 = EmbeddingCache.from_precompute(
        cache_dir=tmp_path, key=key, compute_fn=recompute
    )
    assert called["n"] == 0, "HIT should not invoke compute_fn"
    assert torch.equal(cache2.tensor, written)


def test_cache_key_fingerprint_stable(synthetic_h5: Path) -> None:
    """Same (h5, model_id, mode) → same fingerprint."""
    k1 = CacheKey(h5_path=synthetic_h5, model_id="fake", mode="cls")
    k2 = CacheKey(h5_path=synthetic_h5, model_id="fake", mode="cls")
    assert k1.fingerprint() == k2.fingerprint()


def test_cache_key_model_id_differs(synthetic_h5: Path) -> None:
    """Different model_id → different fingerprint."""
    k1 = CacheKey(h5_path=synthetic_h5, model_id="fake-a", mode="cls")
    k2 = CacheKey(h5_path=synthetic_h5, model_id="fake-b", mode="cls")
    assert k1.fingerprint() != k2.fingerprint()


def test_cache_miss_returns_none(tmp_path: Path, synthetic_h5: Path) -> None:
    key = CacheKey(h5_path=synthetic_h5, model_id="fake", mode="cls")
    assert EmbeddingCache.load(cache_dir=tmp_path, key=key) is None


def test_cache_key_mode_differs(synthetic_h5: Path) -> None:
    """Different encoding mode → different fingerprint (a cls cache must not be
    served for a patch request and vice versa)."""
    k_cls = CacheKey(h5_path=synthetic_h5, model_id="fake", mode="cls")
    k_patch = CacheKey(h5_path=synthetic_h5, model_id="fake", mode="patch2")
    assert k_cls.fingerprint() != k_patch.fingerprint()


def test_cache_load_missing_h5_returns_none(tmp_path: Path) -> None:
    """A gone source HDF5 → the fingerprint (size+mtime) can't be computed →
    load() returns None (a miss), never raises FileNotFoundError."""
    key = CacheKey(h5_path=tmp_path / "gone.h5", model_id="fake", mode="cls")
    assert EmbeddingCache.load(cache_dir=tmp_path, key=key) is None


def test_cache_corrupt_sidecar_returns_none(
    tmp_path: Path, synthetic_h5: Path
) -> None:
    """A truncated/corrupt .json sidecar reads as a miss (recompute), not a
    JSONDecodeError."""
    key = CacheKey(h5_path=synthetic_h5, model_id="fake", mode="cls")
    written = torch.zeros(50, 8)
    EmbeddingCache.from_precompute(
        cache_dir=tmp_path, key=key,
        compute_fn=lambda: (written, {"m": 1}), verbose=False,
    )
    (tmp_path / key.meta_filename()).write_text("{ not valid json")
    assert EmbeddingCache.load(cache_dir=tmp_path, key=key) is None
