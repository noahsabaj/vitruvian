# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""DINOv3 precompute entry points used by ``vit-train``.

Wraps :class:`EmbeddingCache.from_precompute` with the correct
:class:`CacheKey` + compute-fn for each supported latent shape. Running
either function:

1. Loads the frozen DINOv3 backbone.
2. Compiles it (``torch.compile(mode="reduce-overhead")``).
3. Encodes every frame in the HDF5 under ``bf16_autocast``, streaming
   pixel batches from disk so we never hold the full tensor in RAM.
4. Writes a SHA-keyed ``.pt`` + ``.json`` next to each other in
   ``cache_dir``.
5. Frees the DINOv3 VRAM (344 MB) so the subsequent training run gets
   the full budget.
6. Re-loads the tensor via ``mmap=True`` for working-set RAM
   proportional to batch size, not the total (20 GB for patch v5).

These functions replace the near-identical
``precompute_embeddings`` / ``precompute_patch_embeddings`` helpers
that lived in the legacy training scripts.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import h5py
import torch

from vitruvian.data.cache import CacheKey, EmbeddingCache
from vitruvian.models.backbones import (
    DEFAULT_DINOV3_ID,
    DINOv3ClsBackbone,
    DINOv3PatchBackbone,
)
from vitruvian.utils.compile_utils import bf16_autocast, compile_model


def _stream_encode(
    h5_path: Path,
    encode: Callable[[torch.Tensor], torch.Tensor],
    *,
    out_shape: tuple[int, ...],
    out_dtype: torch.dtype,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Run ``encode(batch)`` over the HDF5 pixel array in ``batch_size``
    chunks. Returns a host-memory tensor of shape ``out_shape``.

    ``encode(batch)`` must accept ``(B, 1, 3, H, W)`` and return
    ``(B, 1, ...)`` (the backbone's native per-frame shape). Trailing
    dims are unsqueezed back before writing into ``out``.
    """
    with h5py.File(h5_path, "r") as f:
        pixels_ds = f["pixels"]
        n_total = int(pixels_ds.shape[0])
        if out_shape[0] != n_total:
            raise RuntimeError(
                f"out_shape rows {out_shape[0]} != HDF5 rows {n_total}"
            )
        out = torch.empty(*out_shape, dtype=out_dtype)
        cursor = 0
        t0 = time.perf_counter()
        with torch.no_grad(), bf16_autocast():
            while cursor < n_total:
                end = min(cursor + batch_size, n_total)
                batch_np = pixels_ds[cursor:end]
                batch = (
                    torch.from_numpy(batch_np)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                    .unsqueeze(1)
                    .to(device)
                )
                emb = encode(batch).squeeze(1)
                out[cursor:end] = emb.to(out_dtype).cpu()
                cursor = end
                if cursor % (batch_size * 100) == 0 or cursor == n_total:
                    elapsed = time.perf_counter() - t0
                    print(
                        f"  encoded {cursor}/{n_total}  "
                        f"({cursor / max(elapsed, 1e-6):.0f} fps)"
                    )
    return out


def _build_cls_compute_fn(
    h5_path: Path,
    model_id: str,
    batch_size: int,
    device: str,
    dtype: torch.dtype,
) -> Callable[[], tuple[torch.Tensor, dict[str, Any]]]:
    def _compute() -> tuple[torch.Tensor, dict[str, Any]]:
        backbone = DINOv3ClsBackbone(
            model_id=model_id, device=device, dtype=dtype
        )
        backbone.eval()
        backbone.dinov3 = compile_model(
            backbone.dinov3, mode="reduce-overhead"
        )
        n_total = _hdf5_n_total(h5_path)
        out = _stream_encode(
            h5_path,
            backbone.encode,
            out_shape=(n_total, backbone.output_dim),
            out_dtype=torch.float32,
            batch_size=batch_size,
            device=device,
        )
        metadata = {
            "h5_path": str(h5_path),
            "model_id": model_id,
            "n_total": n_total,
            "dim": int(backbone.output_dim),
        }
        del backbone
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return out, metadata

    return _compute


def _build_patch_compute_fn(
    h5_path: Path,
    model_id: str,
    spatial_stride: int,
    batch_size: int,
    device: str,
    dtype: torch.dtype,
) -> Callable[[], tuple[torch.Tensor, dict[str, Any]]]:
    def _compute() -> tuple[torch.Tensor, dict[str, Any]]:
        backbone = DINOv3PatchBackbone(
            model_id=model_id,
            device=device,
            dtype=dtype,
            spatial_stride=spatial_stride,
        )
        backbone.eval()
        backbone.dinov3 = compile_model(
            backbone.dinov3, mode="reduce-overhead"
        )
        n_patches = backbone.n_patches
        patch_dim = backbone.output_dim
        n_total = _hdf5_n_total(h5_path)
        out = _stream_encode(
            h5_path,
            backbone.encode,
            out_shape=(n_total, n_patches, patch_dim),
            out_dtype=torch.float16,
            batch_size=batch_size,
            device=device,
        )
        metadata = {
            "h5_path": str(h5_path),
            "model_id": model_id,
            "spatial_stride": spatial_stride,
            "n_total": n_total,
            "n_patches": int(n_patches),
            "patch_dim": int(patch_dim),
        }
        del backbone
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return out, metadata

    return _compute


def _hdf5_n_total(h5_path: Path) -> int:
    with h5py.File(h5_path, "r") as f:
        return int(f["pixels"].shape[0])


def build_cls_cache(
    h5_path: Path | str,
    cache_dir: Path | str,
    *,
    model_id: str = DEFAULT_DINOV3_ID,
    batch_size: int = 256,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> EmbeddingCache:
    """HIT returns the mmap view; MISS runs DINOv3 over all frames
    (compiled + BF16) and writes the ``(N, D)`` fp32 cache."""
    key = CacheKey(
        h5_path=Path(h5_path), model_id=model_id, mode="cls"
    )
    return EmbeddingCache.from_precompute(
        cache_dir=Path(cache_dir),
        key=key,
        compute_fn=_build_cls_compute_fn(
            h5_path=Path(h5_path),
            model_id=model_id,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        ),
    )


def build_patch_cache(
    h5_path: Path | str,
    cache_dir: Path | str,
    *,
    model_id: str = DEFAULT_DINOV3_ID,
    spatial_stride: int = 2,
    batch_size: int = 256,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> EmbeddingCache:
    """HIT returns the mmap view; MISS runs DINOv3 over all frames
    (compiled + BF16) and writes the ``(N, n_patches, patch_dim)`` fp16
    cache — ~20 GB for 267k frames at 7×7 patches."""
    key = CacheKey(
        h5_path=Path(h5_path),
        model_id=model_id,
        mode=f"patch{spatial_stride}",
    )
    return EmbeddingCache.from_precompute(
        cache_dir=Path(cache_dir),
        key=key,
        compute_fn=_build_patch_compute_fn(
            h5_path=Path(h5_path),
            model_id=model_id,
            spatial_stride=spatial_stride,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        ),
    )


__all__ = ["build_cls_cache", "build_patch_cache"]
