# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""``vit-train`` — JEPA training entry point.

Usage::

    uv run vit-train configs/train/jepa_v5.yaml --override trainer.lr=3e-5

Takes a YAML config that names the JEPA shape (via the
:mod:`vitruvian.models.registry` keys), the dataset, the loss coeffs,
and the trainer knobs. Wires everything through library primitives
with no ad-hoc plumbing.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from vitruvian.data import (
    CacheKey,
    EmbeddingCache,
    G1EmbSeqDataset,
    G1PatchSeqDataset,
    build_cls_cache,
    build_patch_cache,
    episode_aware_split,
)
from vitruvian.models import build_jepa
from vitruvian.training import JEPATrainer, TrainerConfig, prediction_loss
from vitruvian.utils import load_config


def _build_cache(
    data_cfg: dict,
    device: str,
    *,
    skip_precompute: bool = False,
) -> tuple[EmbeddingCache, str]:
    """Load (or precompute) the per-frame embedding cache.

    On cache HIT: mmap-loads in milliseconds regardless of file size.

    On cache MISS: loads frozen DINOv3, compiles it, streams the HDF5
    through with bf16 autocast, writes the cache, releases the backbone
    VRAM, and reopens mmap. End-to-end for 267k frames × patches:
    roughly 6 min on an 8 GB 4060 Ti.

    Set ``skip_precompute=True`` to error out on MISS instead (useful
    for debugging / CI where we don't want to accidentally kick off a
    long precompute).
    """
    h5_path = Path(data_cfg["h5"]).expanduser()
    cache_dir = Path(data_cfg["cache_dir"]).expanduser()
    mode = data_cfg.get("cache_mode", "cls")
    model_id = data_cfg.get(
        "model_id", "facebook/dinov3-vitb16-pretrain-lvd1689m"
    )
    key = CacheKey(h5_path=h5_path, model_id=model_id, mode=mode)

    cache = EmbeddingCache.load(cache_dir=cache_dir, key=key)
    if cache is not None:
        return cache, mode

    if skip_precompute:
        raise SystemExit(
            f"cache MISS for {key.filename()} at {cache_dir} and "
            f"--skip-precompute was set; aborting."
        )

    batch_size = int(data_cfg.get("precompute_batch_size", 256))
    print(
        f"[precompute] MISS for {key.filename()}; running DINOv3 over "
        f"{h5_path.name} (mode={mode}, model_id={model_id}, "
        f"batch={batch_size})"
    )
    if mode == "cls":
        cache = build_cls_cache(
            h5_path=h5_path,
            cache_dir=cache_dir,
            model_id=model_id,
            batch_size=batch_size,
            device=device,
        )
    elif mode.startswith("patch"):
        spatial_stride = int(mode[len("patch") :]) if mode != "patch" else 2
        cache = build_patch_cache(
            h5_path=h5_path,
            cache_dir=cache_dir,
            model_id=model_id,
            spatial_stride=spatial_stride,
            batch_size=batch_size,
            device=device,
        )
    else:
        raise ValueError(
            f"unknown cache_mode {mode!r}; expected 'cls' or 'patchN'"
        )
    return cache, mode


def _build_loaders(
    data_cfg: dict,
    cache: EmbeddingCache,
    cache_mode: str,
    *,
    seq_len: int,
    trainer_cfg: TrainerConfig,
    quick: bool,
) -> tuple[DataLoader, DataLoader]:
    h5_path = Path(data_cfg["h5"]).expanduser()
    dataset = (
        G1PatchSeqDataset(h5_path, cache.tensor, seq_len=seq_len)
        if cache_mode.startswith("patch")
        else G1EmbSeqDataset(h5_path, cache.tensor, seq_len=seq_len)
    )

    # Episode-aware split. Sliding windows from one episode overlap by
    # seq_len-1 frames, so a per-window random_split leaks frames across
    # train/val and inflates val_pred_loss (used for model selection).
    # Splitting by whole episode keeps the two sets frame-disjoint.
    train_idx, val_idx = episode_aware_split(
        dataset.ep_offset, dataset.valid_idx,
        val_frac=trainer_cfg.val_frac, seed=trainer_cfg.seed, seq_len=seq_len,
    )

    subsample = int(data_cfg.get("subsample", 0) or 0)
    if quick and subsample == 0:
        subsample = 2000
    if subsample > 0:
        train_idx = train_idx[:subsample]
        val_idx = val_idx[: max(1, int(round(trainer_cfg.val_frac * subsample)))]

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    nw = trainer_cfg.num_workers if not quick else 0
    train_loader = DataLoader(
        train_set,
        batch_size=trainer_cfg.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=trainer_cfg.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a JEPA from a YAML config.")
    ap.add_argument("config", type=Path, help="YAML config path")
    ap.add_argument(
        "--override", "-o",
        action="append",
        default=[],
        help="Dotted override, e.g. trainer.lr=3e-5",
    )
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--quick-debug",
        action="store_true",
        help="1 epoch on 2000 samples for plumbing smoke.",
    )
    ap.add_argument(
        "--skip-precompute",
        action="store_true",
        help="Error on cache MISS instead of running DINOv3 precompute. "
        "Useful for CI or when the cache is known to be warm.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_cfg = cfg["data"]
    cache, cache_mode = _build_cache(
        data_cfg, device=device, skip_precompute=args.skip_precompute
    )
    print(f"[cache]  {cache.describe()}")

    trainer_cfg = TrainerConfig(**cfg.get("trainer", {}))
    if args.quick_debug:
        trainer_cfg.epochs = 1

    loss_cfg = cfg.get("loss", {})
    history_size = int(loss_cfg.get("history_size", 3))
    num_preds = int(loss_cfg.get("num_preds", 6))
    rollout_weight = float(loss_cfg.get("rollout_weight", 1.0))
    # ``reg_weight`` weights the SIGReg isotropic-Gaussian term; accept
    # the legacy ``std_weight`` key as an alias.
    reg_weight = float(
        loss_cfg.get("reg_weight", loss_cfg.get("std_weight", 0.0))
    )
    # Per-frame proprio-conditioning dropout so the model tolerates the
    # proprio-free future of a plan-time rollout.
    proprio_dropout = float(loss_cfg.get("proprio_dropout", 0.5))
    seq_len = history_size + num_preds

    train_loader, val_loader = _build_loaders(
        data_cfg, cache, cache_mode,
        seq_len=seq_len,
        trainer_cfg=trainer_cfg,
        quick=args.quick_debug,
    )

    # Build JEPA. Training reads the precomputed embedding cache, so the
    # backbone forward is never called — load it lazily (both cls and
    # patch) to free the ~344 MB of DINOv3 VRAM for the trainable
    # predictor. ``vit-plan``/``vit-eval`` hydrate it via load_eagerly().
    jepa_cfg = cfg["jepa"]
    jepa_cfg.setdefault("backbone", {}).setdefault("kwargs", {}).update(
        {"lazy": True, "device": device}
    )
    jepa = build_jepa(jepa_cfg).to(device)

    loss_fn = partial(
        prediction_loss,
        history_size=history_size,
        num_preds=num_preds,
        rollout_weight=rollout_weight,
        reg_weight=reg_weight,
        proprio_dropout=proprio_dropout,
    )

    run_name = args.run_name or cfg.get("run_name", "jepa")
    out_dir = (
        Path(args.out_dir).expanduser()
        if args.out_dir is not None
        else Path(cfg.get("out_dir", "~/.vitruvian/run")).expanduser()
    )
    trainer = JEPATrainer(
        jepa=jepa,
        loss_fn=loss_fn,
        cfg=trainer_cfg,
        out_dir=out_dir,
        run_name=run_name,
        jepa_config=jepa_cfg,
    )
    trainer.fit(train_loader, val_loader, device=device)


if __name__ == "__main__":
    main()
