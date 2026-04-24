# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""M4.7 — ``EmbeddingCache``: memory-mapped loader for precomputed
encoder caches (DINOv3 CLS, DINOv3 patches, VF embeddings, etc.).

**Why this module exists.** M4.5 through M4.6 each trained against a
multi-GB precomputed-encoder cache. Three training scripts
(`m4e_train_jepa_v4.py`, `m4e_train_vf_her.py`, `m4f_train_jepa_v5.py`)
loaded those caches with bespoke `torch.load(path, map_location="cpu")`
calls. For the v5 patch cache (20 GB fp16) that eager load:

  * OOM-killed the v5 training run at epoch 2 when `num_workers=2`
    caused Dataloader worker forks to balloon RSS past 32 GB RAM.
  * Forced `num_workers=0`, which throttled epoch 1 throughput by
    ~30% because batch reads block the GPU step.

`torch.load(..., mmap=True)` (PyTorch ≥ 2.1) fixes both: the OS page
cache handles paging, RSS stays O(working set), and Dataloader workers
share the mapped pages. But there are footguns (the returned tensor's
device must be CPU; the cache file must be accessed via random reads,
not writes; stale caches must be invalidated when the source HDF5 or
encoder model changes). This module centralizes all of that.

**Invariants.**

  * Caches are SHA-keyed on (source HDF5 path + size + mtime, encoder
    model ID, encoding-mode). A new HDF5 or a new DINOv3 release
    invalidates.
  * Metadata lives alongside the `.pt` in a `.json` sidecar.
  * The cache tensor is always on CPU; callers move to GPU per batch.
  * ``from_precompute(...)`` is the one-stop entry that either loads
    the cache (HIT) or computes and writes it (MISS).

Callers after M4.7:
  * `scripts/m4e_train_jepa_v4.py:precompute_embeddings` → replaced by
    ``EmbeddingCache.from_precompute(mode="cls")``.
  * `scripts/m4f_train_jepa_v5.py:precompute_patch_embeddings` →
    ``mode="patch7"`` (7×7 subsampled).
  * `scripts/m4e_train_vf_her.py` → ``load`` (never computes, reuses
    the v4/v5 training cache).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


# --------------------------------------------------------------------------
# Metadata + key
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheKey:
    """Everything that uniquely identifies a cache. Two CacheKeys with
    the same fingerprint are considered interchangeable — the cache
    file can be shared."""

    h5_path: Path
    model_id: str
    mode: str  # e.g. "cls", "patch7", "patch14"

    def fingerprint(self) -> str:
        """SHA-256 fingerprint; also depends on the HDF5's size + mtime
        so a replaced file invalidates."""
        h = hashlib.sha256()
        h.update(str(Path(self.h5_path).resolve()).encode())
        h.update(b"\x00")
        h.update(self.model_id.encode())
        h.update(b"\x00")
        h.update(self.mode.encode())
        h.update(b"\x00")
        st = Path(self.h5_path).stat()
        h.update(f"{st.st_size}-{int(st.st_mtime)}".encode())
        return h.hexdigest()[:16]

    def filename(self) -> str:
        return f"{self.mode}_{self.fingerprint()}.pt"

    def meta_filename(self) -> str:
        return f"{self.mode}_{self.fingerprint()}.json"


# --------------------------------------------------------------------------
# The cache object
# --------------------------------------------------------------------------


class EmbeddingCache:
    """Handle to a precomputed encoder cache on disk.

    Use ``.tensor`` to get the memory-mapped CPU tensor. Shape depends
    on ``mode``:
      * ``"cls"``:    (N, D)
      * ``"patch7"``: (N, 49, D)
      * ``"patch14"``: (N, 196, D)
    """

    def __init__(
        self,
        path: Path,
        meta_path: Path,
        tensor: torch.Tensor,
        metadata: dict[str, Any],
    ) -> None:
        self.path = Path(path)
        self.meta_path = Path(meta_path)
        self._tensor = tensor
        self.metadata = metadata

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, cache_dir: Path, key: CacheKey) -> "EmbeddingCache | None":
        """Try to load the cache. Returns ``None`` on miss — caller
        decides whether to compute it."""
        cache_dir = Path(cache_dir)
        pt_path = cache_dir / key.filename()
        meta_path = cache_dir / key.meta_filename()
        if not (pt_path.exists() and meta_path.exists()):
            return None
        # mmap=True keeps RSS ~O(batch) instead of reading the whole file.
        tensor = torch.load(
            pt_path, map_location="cpu", weights_only=True, mmap=True
        )
        with meta_path.open("r") as f:
            metadata = json.load(f)
        return cls(pt_path, meta_path, tensor, metadata)

    @classmethod
    def from_precompute(
        cls,
        cache_dir: Path,
        key: CacheKey,
        compute_fn: Callable[[], tuple[torch.Tensor, dict[str, Any]]],
        *,
        verbose: bool = True,
    ) -> "EmbeddingCache":
        """Load if present (HIT), otherwise call ``compute_fn`` and
        write the cache (MISS), then load the mmap view.

        ``compute_fn`` is invoked with no args on MISS and must return
        ``(tensor, metadata_dict)``. ``tensor`` is saved to disk and
        then re-opened mmap so the returned cache has the memory-map
        view (not the eager tensor held by ``compute_fn``).
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        existing = cls.load(cache_dir, key)
        if existing is not None:
            if verbose:
                print(f"[cache]  HIT   {existing.path.name}")
            return existing

        if verbose:
            print(f"[cache]  MISS  computing {key.filename()}...")
        t0 = time.perf_counter()
        tensor, metadata = compute_fn()
        elapsed = time.perf_counter() - t0
        pt_path = cache_dir / key.filename()
        meta_path = cache_dir / key.meta_filename()
        torch.save(tensor, pt_path)
        with meta_path.open("w") as f:
            json.dump(metadata, f, indent=2)
        if verbose:
            size_gb = pt_path.stat().st_size / 1e9
            print(
                f"[cache]  wrote {pt_path.name}  "
                f"({elapsed:.1f}s compute, {size_gb:.2f} GB on disk)"
            )
        # Re-load via mmap so the tensor held by the caller doesn't eat RAM.
        del tensor
        return cls.load(cache_dir, key)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def tensor(self) -> torch.Tensor:
        """The memory-mapped CPU tensor. Safe to slice; slices trigger
        page reads on demand."""
        return self._tensor

    def __len__(self) -> int:
        return self._tensor.shape[0]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._tensor.shape)

    @property
    def dtype(self) -> torch.dtype:
        return self._tensor.dtype

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def size_bytes(self) -> int:
        """Number of bytes backing the mmap (i.e., file size on disk)."""
        return self.path.stat().st_size

    def describe(self) -> str:
        return (
            f"EmbeddingCache(path={self.path.name}, "
            f"shape={tuple(self._tensor.shape)}, dtype={self._tensor.dtype}, "
            f"file={self.size_bytes() / 1e9:.2f} GB)"
        )


__all__ = ["CacheKey", "EmbeddingCache"]
