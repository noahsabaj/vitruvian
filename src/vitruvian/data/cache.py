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
    invalidates. The encoder *dtype* is intentionally NOT in the key
    (caches are always built at the builder's default precision); it is
    recorded in the metadata sidecar for inspection. mtime is truncated
    to whole seconds, so a same-size in-place replacement within one
    second is a (rare) stale-HIT footgun.
  * Metadata lives alongside the `.pt` in a `.json` sidecar; both are
    written atomically (temp file + rename) so a crash mid-write leaves
    a clean MISS, not a truncated sidecar.
  * The cache tensor is always on CPU; callers move to GPU per batch.
  * ``from_precompute(...)`` is the one-stop entry that either loads
    the cache (HIT) or computes and writes it (MISS).

Entry points now live in :mod:`vitruvian.data.precompute`
(``build_cls_cache`` / ``build_patch_cache``, driven by ``vit-train``);
``vit-rollout`` uses :meth:`EmbeddingCache.load` to reuse the training
cache without ever recomputing.
"""

from __future__ import annotations

import hashlib
import json
import os
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

    Use ``.tensor`` to get the memory-mapped CPU tensor. The mode string
    is ``"cls"`` or ``f"patch{spatial_stride}"`` (see
    :func:`vitruvian.data.precompute.build_patch_cache`). Shape depends
    on ``mode``:
      * ``"cls"``:    (N, D)
      * ``"patch2"``: (N, 49, D)   — stride-2 subsample of the 14×14 grid
      * ``"patch1"``: (N, 196, D)  — full 14×14 grid (no subsampling)
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
        decides whether to compute it.

        Treated as a MISS (``None``), never an exception: the source HDF5
        being gone (the key can't be computed without its size+mtime) and a
        truncated/corrupt sidecar (e.g. from a crash mid-write in an older
        build). This keeps the documented "returns None on miss" contract for
        the load-only flows (``vit-rollout``, the VF trainer)."""
        cache_dir = Path(cache_dir)
        try:
            pt_path = cache_dir / key.filename()
            meta_path = cache_dir / key.meta_filename()
        except FileNotFoundError:
            # Source HDF5 missing → fingerprint (size+mtime) can't be computed,
            # so the cache is unfindable — a miss, not a crash.
            return None
        if not (pt_path.exists() and meta_path.exists()):
            return None
        try:
            with meta_path.open("r") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None  # corrupt/half-written sidecar → recompute
        # mmap=True keeps RSS ~O(batch) instead of reading the whole file.
        tensor = torch.load(
            pt_path, map_location="cpu", weights_only=True, mmap=True
        )
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
        # Write to temp files then atomically rename, sidecar LAST: a crash
        # can't leave a truncated .json or a .pt with no metadata (load()
        # requires BOTH to exist, so a partial write reads as a clean miss).
        pt_tmp = pt_path.with_name(pt_path.name + ".tmp")
        meta_tmp = meta_path.with_name(meta_path.name + ".tmp")
        torch.save(tensor, pt_tmp)
        with meta_tmp.open("w") as f:
            json.dump(metadata, f, indent=2)
        os.replace(pt_tmp, pt_path)
        os.replace(meta_tmp, meta_path)
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
