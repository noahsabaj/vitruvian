#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""M4.5 — train JEPAv4 (frozen DINOv3 + trainable proprio + ARPredictor)
on the diverse G1 expert dataset.

Pipeline:
  1. Load frozen DINOv3 ViT-B/16. Precompute the 768-D CLS embedding
     for every frame in the HDF5 once → cache as torch tensor file.
  2. Train predictor + proprio encoder + action encoder on the cached
     embeddings with 1-step TF MSE + k-step rollout MSE up to
     num_preds=6, following Terver et al. (arXiv:2512.24497).

The DINOv3 forward is NEVER called during training itself — only once
during precompute. Training step is pure predictor forward/backward
plus a small proprio MLP, so 3 epochs over 270k transitions fits in
~1-2h even on an 8 GB GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "le-wm"))

from vitruvian.hwm.backbone_dinov3 import DEFAULT_DINOV3_ID, DINOv3Backbone  # noqa: E402
from vitruvian.hwm.jepa_v4 import JEPAv4  # noqa: E402


# --------------------------------------------------------------------------
# Precompute cache: DINOv3 CLS embeddings for every frame
# --------------------------------------------------------------------------


def _cache_key(h5_path: Path, model_id: str) -> str:
    h = hashlib.sha256()
    h.update(str(h5_path.resolve()).encode())
    h.update(b"\x00")
    h.update(model_id.encode())
    h.update(b"\x00")
    # Include file size + mtime for invalidation on HDF5 change.
    st = h5_path.stat()
    h.update(f"{st.st_size}-{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def precompute_embeddings(
    h5_path: Path,
    cache_dir: Path,
    *,
    model_id: str = DEFAULT_DINOV3_ID,
    batch_size: int = 128,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> tuple[Path, torch.Tensor]:
    """Encode every frame in the HDF5 with frozen DINOv3, cache result.

    Returns (cache_path, embeddings tensor of shape (N_total, 768) float32).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(h5_path, model_id)
    cache_path = cache_dir / f"dinov3_cls_{key}.pt"
    meta_path = cache_dir / f"dinov3_cls_{key}.json"

    if cache_path.exists() and meta_path.exists():
        print(f"[cache]  HIT  {cache_path.name}")
        embs = torch.load(cache_path, map_location="cpu", weights_only=True)
        return cache_path, embs

    print(f"[cache]  MISS — encoding {h5_path.name} with {model_id}...")
    backbone = DINOv3Backbone(model_id=model_id, device=device, dtype=dtype)
    backbone.eval()

    with h5py.File(h5_path, "r") as f:
        pixels_ds = f["pixels"]  # (N, H, W, 3) uint8
        n_total = int(pixels_ds.shape[0])

        out = torch.empty(n_total, backbone.output_dim, dtype=torch.float32)
        t0 = time.perf_counter()
        cursor = 0
        with torch.no_grad():
            while cursor < n_total:
                end = min(cursor + batch_size, n_total)
                batch_np = pixels_ds[cursor:end]  # (B, H, W, 3) uint8
                batch = (
                    torch.from_numpy(batch_np)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                    .unsqueeze(1)
                    .to(device)
                )  # (B, 1, 3, H, W) uint8
                emb = backbone.encode(batch).squeeze(1).float().cpu()  # (B, 768)
                out[cursor:end] = emb
                cursor = end
                if cursor % (batch_size * 100) == 0 or cursor == n_total:
                    elapsed = time.perf_counter() - t0
                    print(
                        f"  encoded {cursor}/{n_total}  "
                        f"({cursor / max(elapsed, 1e-6):.0f} fps)"
                    )

    torch.save(out, cache_path)
    with meta_path.open("w") as mf:
        json.dump(
            {"h5_path": str(h5_path), "model_id": model_id, "n_total": n_total, "dim": int(backbone.output_dim)},
            mf,
            indent=2,
        )
    elapsed = time.perf_counter() - t0
    print(f"[cache]  wrote {cache_path.name}  ({elapsed:.1f}s total)")

    # Free DINOv3 weights from VRAM before training.
    del backbone
    torch.cuda.empty_cache()
    return cache_path, out


# --------------------------------------------------------------------------
# Dataset of (embedding, proprio, action) sequences
# --------------------------------------------------------------------------


class G1EmbSeqDataset(Dataset):
    """Yields dict with:
        "emb":     (seq_len, D_emb)      float32
        "proprio": (seq_len, D_prop)     float32
        "action":  (seq_len, action_dim) float32

    where seq_len = history_size + num_preds. Samples are constructed
    per-sample on-the-fly by indexing pre-built start/ep arrays; goals
    are sampled uniformly from "any state in the same episode that's
    seq_len frames ahead of the start".

    Pixel data is NOT loaded — we use the precomputed DINOv3 embeddings.
    Proprio and actions are cached into RAM at __init__ (~100 MB for
    270k × 132 float32).
    """

    def __init__(
        self,
        h5_path: Path,
        emb_cache: torch.Tensor,
        *,
        seq_len: int,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.seq_len = int(seq_len)
        with h5py.File(self.h5_path, "r") as f:
            ep_offset = f["ep_offset"][:].astype(np.int64)
            ep_len = f["ep_len"][:].astype(np.int64)
            n_total = int(f["pixels"].shape[0])
            # Cache all proprio + action into host RAM.
            self.proprio = torch.from_numpy(f["proprio"][:]).float()
            self.action = torch.from_numpy(f["action"][:]).float()
        assert emb_cache.shape[0] == n_total, (
            f"emb cache rows {emb_cache.shape[0]} != H5 rows {n_total}"
        )
        self.emb = emb_cache
        self.ep_offset = ep_offset
        self.ep_len = ep_len

        # Valid sample starts: offset o in each episode such that
        # [o, o + seq_len) stays within that episode.
        valid = []
        for o, L in zip(ep_offset, ep_len):
            for t in range(int(o), int(o + L - seq_len + 1)):
                valid.append(t)
        self.valid_idx = np.asarray(valid, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.valid_idx)

    def __getitem__(self, idx: int) -> dict:
        t0 = int(self.valid_idx[idx])
        s = slice(t0, t0 + self.seq_len)
        return {
            "emb": self.emb[s],          # (seq_len, D_emb)
            "proprio": self.proprio[s],  # (seq_len, D_prop)
            "action": self.action[s],    # (seq_len, action_dim)
        }


# --------------------------------------------------------------------------
# Loss — Terver recipe: 1-step TF MSE + k-step rollout MSE
# --------------------------------------------------------------------------


def compute_loss(
    model: JEPAv4,
    batch: dict,
    *,
    history_size: int,
    num_preds: int,
    rollout_weight: float,
) -> dict:
    """Mirror of LeWM's ``lejepa_forward`` trimmed for v4.

    batch["emb"]:     (B, T, D_emb) — precomputed DINOv3 CLS; CONST grad.
    batch["proprio"]: (B, T, D_prop)
    batch["action"]:  (B, T, A_dim) — raw actions at each frame.

    We add the proprio MLP output to the (frozen) CLS to form the
    predictor's input state. All other modules train.
    """
    emb_vis = batch["emb"]       # (B, T, D)
    proprio = batch["proprio"]   # (B, T, D_prop)
    action = batch["action"]     # (B, T, A_dim)

    # State fusion (trainable proprio branch).
    prop_emb = model.proprio_encoder(proprio) if model.proprio_encoder is not None else 0.0
    emb = emb_vis + prop_emb  # (B, T, D)

    # Action encoding (trainable).
    # At frameskip=1, each frame's action is the raw 29-D vector;
    # Embedder wants input_dim = action_dim * frameskip = 29.
    act_emb = model.action_encoder(action)  # (B, T, D)

    ctx_len = history_size
    n_preds = num_preds
    ctx_emb = emb[:, :ctx_len]  # (B, ctx_len, D)
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds : n_preds + ctx_len]  # (B, ctx_len, D)

    pred_emb = model.predict(ctx_emb, ctx_act)  # (B, ctx_len, D)
    pred_loss = (pred_emb - tgt_emb).pow(2).mean()

    rollout_losses: list[torch.Tensor] = []
    if rollout_weight > 0 and n_preds >= 2:
        rolling_emb = ctx_emb
        for k in range(1, n_preds):
            rolling_emb = torch.cat(
                [rolling_emb[:, 1:], pred_emb[:, -1:]], dim=1
            )
            rolling_act = act_emb[:, k : k + ctx_len]
            pred_emb = model.predict(rolling_emb, rolling_act)
            tgt_k = emb[:, k + ctx_len - 1 : k + ctx_len]  # (B, 1, D)
            rollout_losses.append((pred_emb[:, -1:] - tgt_k).pow(2).mean())
        rollout_loss = sum(rollout_losses) / max(1, len(rollout_losses))
    else:
        rollout_loss = torch.zeros((), device=emb.device)

    total = pred_loss + rollout_weight * rollout_loss
    return {
        "loss": total,
        "pred_loss": pred_loss.detach(),
        "rollout_loss": rollout_loss.detach()
        if torch.is_tensor(rollout_loss)
        else torch.zeros((), device=emb.device),
        "n_rollout_steps": len(rollout_losses),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def cosine_lr_factor(step: int, total: int, warmup: int, peak: float, floor: float) -> float:
    if step < warmup:
        return float(step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (floor + (peak - floor) * cos) / peak


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--h5",
        type=Path,
        default=Path.home() / ".stable_worldmodel" / "g1_diverse_v1.h5",
    )
    ap.add_argument(
        "--dinov3-id",
        type=str,
        default=DEFAULT_DINOV3_ID,
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".vitruvian" / "m4e_v4" / "cache",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path.home() / ".vitruvian" / "m4e_v4",
    )
    ap.add_argument("--run-name", type=str, default="jepa_v4")

    # Model.
    ap.add_argument("--history-size", type=int, default=3)
    ap.add_argument("--num-preds", type=int, default=6)
    ap.add_argument("--proprio-hidden", type=int, default=256)
    ap.add_argument("--action-frameskip", type=int, default=1)
    ap.add_argument("--action-smoothed-dim", type=int, default=10)
    ap.add_argument("--predictor-depth", type=int, default=6)
    ap.add_argument("--predictor-heads", type=int, default=16)
    ap.add_argument("--predictor-mlp-dim", type=int, default=3072)
    ap.add_argument("--predictor-dropout", type=float, default=0.1)

    # Training.
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-floor", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--rollout-weight", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--precompute-batch", type=int, default=128,
        help="Batch size for DINOv3 precompute pass (VRAM-bound).",
    )
    ap.add_argument(
        "--quick-debug",
        action="store_true",
        help="1 epoch over 2000 samples for plumbing smoke test.",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"--- M4.5  JEPAv4 training (DINOv3 + proprio + ARPredictor) ---")
    print(f"device:          {device}")
    print(f"h5:              {args.h5}")
    print(f"dinov3-id:       {args.dinov3_id}")
    print(f"cache-dir:       {args.cache_dir}")
    print(f"out-dir:         {args.out_dir}")
    print(f"history-size:    {args.history_size}  num-preds: {args.num_preds}")
    print(f"batch-size:      {args.batch_size}  epochs: {args.epochs}  lr: {args.lr}")
    print(f"quick-debug:     {args.quick_debug}")
    print()

    # ---- 1. Precompute DINOv3 encodings ----
    cache_path, emb_cache = precompute_embeddings(
        args.h5,
        args.cache_dir,
        model_id=args.dinov3_id,
        batch_size=args.precompute_batch,
        device=device,
    )

    # ---- 2. Build dataset ----
    seq_len = args.history_size + args.num_preds
    dataset = G1EmbSeqDataset(args.h5, emb_cache, seq_len=seq_len)
    print(f"[data]  samples (valid starts): {len(dataset)}  seq_len: {seq_len}")

    if args.quick_debug:
        # Subsample for fast plumbing check.
        from torch.utils.data import Subset
        n_smoke = min(2000, len(dataset))
        dataset = Subset(dataset, list(range(n_smoke)))
        print(f"[quick-debug] truncated to {len(dataset)} samples")
        args.epochs = 1

    # Train/val split.
    rnd_gen = torch.Generator().manual_seed(args.seed)
    n_total = len(dataset)
    n_val = max(1, int(args.val_frac * n_total))
    n_train = n_total - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=rnd_gen
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if not args.quick_debug else 0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers if not args.quick_debug else 0,
        pin_memory=True,
        drop_last=False,
    )
    print(f"[data]  train: {len(train_set)}  val: {len(val_set)}")

    # ---- 3. Build JEPAv4 ----
    proprio_dim = dataset[0]["proprio"].shape[-1] if len(dataset) > 0 else 103
    action_dim = dataset[0]["action"].shape[-1] if len(dataset) > 0 else 29

    # NOTE: constructing JEPAv4 here will reload DINOv3 into VRAM —
    # unavoidable because rollout/inference paths need it. But we're
    # done with the precompute pass now so this is fine.
    model = JEPAv4(
        dinov3_model_id=args.dinov3_id,
        proprio_dim=int(proprio_dim),
        proprio_hidden=args.proprio_hidden,
        action_dim=int(action_dim),
        action_frameskip=args.action_frameskip,
        action_smoothed_dim=args.action_smoothed_dim,
        predictor_num_frames=args.history_size,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        predictor_mlp_dim=args.predictor_mlp_dim,
        predictor_dropout=args.predictor_dropout,
        device=device,
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen = [p for p in model.parameters() if not p.requires_grad]
    print(f"[model]  trainable params: {sum(p.numel() for p in trainable):,}")
    print(f"[model]  frozen params:    {sum(p.numel() for p in frozen):,}  "
          f"(expected ≈ 85,660,416 for DINOv3 ViT-B)")

    opt = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup = min(args.warmup_steps, total_steps // 10)

    # ---- 4. Train ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    step = 0
    t_start = time.perf_counter()

    jepa_config = dict(
        dinov3_model_id=args.dinov3_id,
        proprio_dim=int(proprio_dim),
        proprio_hidden=args.proprio_hidden,
        action_dim=int(action_dim),
        action_frameskip=args.action_frameskip,
        action_smoothed_dim=args.action_smoothed_dim,
        predictor_num_frames=args.history_size,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        predictor_mlp_dim=args.predictor_mlp_dim,
        predictor_dropout=args.predictor_dropout,
    )

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        # Keep DINOv3 in eval mode even in model.train() so its dropouts don't fire.
        model.backbone.dinov3.eval()
        tr_pred, tr_roll = [], []
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            lr_mult = cosine_lr_factor(step, total_steps, warmup, args.lr, args.lr_floor)
            for pg in opt.param_groups:
                pg["lr"] = args.lr * lr_mult

            info = compute_loss(
                model,
                batch,
                history_size=args.history_size,
                num_preds=args.num_preds,
                rollout_weight=args.rollout_weight,
            )
            loss = info["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tr_pred.append(float(info["pred_loss"]))
            tr_roll.append(float(info["rollout_loss"]))
            step += 1

        # Val
        model.eval()
        with torch.no_grad():
            va_pred, va_roll = [], []
            for batch in val_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                info = compute_loss(
                    model,
                    batch,
                    history_size=args.history_size,
                    num_preds=args.num_preds,
                    rollout_weight=args.rollout_weight,
                )
                va_pred.append(float(info["pred_loss"]))
                va_roll.append(float(info["rollout_loss"]))
        tr_pred_m = sum(tr_pred) / max(1, len(tr_pred))
        tr_roll_m = sum(tr_roll) / max(1, len(tr_roll))
        va_pred_m = sum(va_pred) / max(1, len(va_pred))
        va_roll_m = sum(va_roll) / max(1, len(va_roll))
        elapsed = time.perf_counter() - t_start
        cur_lr = opt.param_groups[0]["lr"]
        print(
            f"[epoch {epoch:>2}/{args.epochs}]  "
            f"train_pred={tr_pred_m:.4f}  train_roll={tr_roll_m:.4f}  "
            f"val_pred={va_pred_m:.4f}  val_roll={va_roll_m:.4f}  "
            f"lr={cur_lr:.2e}  ({elapsed:.1f}s)"
        )

        # Checkpoint every epoch; also save best-val.
        # Persist only TRAINABLE state (predictor, proprio_encoder,
        # action_encoder) — DINOv3 is reloaded from HF at load time, so
        # storing its 344 MB of weights would bloat every epoch ckpt
        # for no benefit.
        trainable_state = {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
            if not k.startswith("backbone.dinov3.")
        }
        ckpt = {
            "config": jepa_config,
            "state_dict": trainable_state,
            "epoch": epoch,
            "train_pred": tr_pred_m,
            "val_pred": va_pred_m,
            "val_rollout": va_roll_m,
            "run_name": args.run_name,
        }
        latest = args.out_dir / "latest.pt"
        torch.save(ckpt, args.out_dir / f"epoch_{epoch:03d}.pt")
        torch.save(ckpt, latest)
        if va_pred_m < best_val:
            best_val = va_pred_m
            torch.save(ckpt, args.out_dir / "best.pt")

    print(f"\n=== M4.5 JEPAv4 done in {time.perf_counter() - t_start:.1f}s ===")
    print(f"best val_pred: {best_val:.4f}")
    print(f"artifacts in:  {args.out_dir}")


if __name__ == "__main__":
    main()
