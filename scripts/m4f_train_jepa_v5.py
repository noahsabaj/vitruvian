#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""M4.6 — train JEPAv5 (frozen DINOv3 7×7 patches + trainable
projection + proprio MLP + PatchARPredictor).

Fork of ``m4e_train_jepa_v4.py`` with two changes:

- **Patch precompute cache** stores ``(N, 49, 768)`` fp16 — ~20 GB for
  267k frames — instead of v4's ``(N, 768)`` fp32.
- **Loss is per-patch L2 MSE + k-step patch rollout MSE.** The training
  objective structure mirrors Terver et al. Fig 3b and LeWM v3
  (``num_preds=6`` supervision).

Trainable modules: `patch_proj` (768→256), `proprio_encoder`,
`action_encoder`, `predictor`. DINOv3 backbone stays frozen.
"""

from __future__ import annotations

import argparse
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

from vitruvian.hwm.backbone_dinov3_patches import (  # noqa: E402
    DEFAULT_DINOV3_ID,
    DINOv3PatchBackbone,
)
from vitruvian.hwm.cache import CacheKey, EmbeddingCache  # noqa: E402
from vitruvian.hwm.compile_utils import bf16_autocast, compile_model  # noqa: E402
from vitruvian.hwm.jepa_v5 import JEPAv5  # noqa: E402


# --------------------------------------------------------------------------
# Patch precompute cache — uses M4.7 EmbeddingCache (mmap load on HIT).
# --------------------------------------------------------------------------


def _build_patch_compute_fn(
    h5_path: Path,
    model_id: str,
    spatial_stride: int,
    batch_size: int,
    device: str,
    dtype: torch.dtype = torch.float16,
):
    """Factory for the compute callback ``EmbeddingCache.from_precompute``
    invokes on MISS. Encodes the full HDF5 under BF16 autocast with a
    compiled DINOv3 forward — ~3× faster than the M4.6 path.
    """

    def _compute() -> tuple[torch.Tensor, dict]:
        backbone = DINOv3PatchBackbone(
            model_id=model_id,
            device=device,
            dtype=dtype,
            spatial_stride=spatial_stride,
        )
        backbone.eval()
        # Compile the underlying DINOv3 ViT so the 267k-frame encode pass
        # hits fused kernels (M4.7 speedup #7).
        backbone.dinov3 = compile_model(backbone.dinov3, mode="reduce-overhead")
        n_patches = backbone.n_patches
        patch_dim = backbone.output_dim

        with h5py.File(h5_path, "r") as f:
            pixels_ds = f["pixels"]
            n_total = int(pixels_ds.shape[0])
            out = torch.empty(n_total, n_patches, patch_dim, dtype=torch.float16)
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
                    emb = backbone.encode(batch).squeeze(1)
                    out[cursor:end] = emb.to(torch.float16).cpu()
                    cursor = end
                    if cursor % (batch_size * 100) == 0 or cursor == n_total:
                        elapsed = time.perf_counter() - t0
                        print(
                            f"  encoded {cursor}/{n_total}  "
                            f"({cursor / max(elapsed, 1e-6):.0f} fps)"
                        )

        metadata = {
            "h5_path": str(h5_path),
            "model_id": model_id,
            "spatial_stride": spatial_stride,
            "n_total": n_total,
            "n_patches": int(n_patches),
            "patch_dim": int(patch_dim),
        }
        # Release DINOv3 VRAM before training starts.
        del backbone
        torch.cuda.empty_cache()
        return out, metadata

    return _compute


def precompute_patch_embeddings(
    h5_path: Path,
    cache_dir: Path,
    *,
    model_id: str = DEFAULT_DINOV3_ID,
    spatial_stride: int = 2,
    batch_size: int = 256,
    device: str = "cuda",
) -> EmbeddingCache:
    """Obtain the memory-mapped 7×7 patch cache for ``h5_path``.

    HIT returns the mmap view instantly (working-set RAM ≈ per-batch,
    not the 20 GB total). MISS runs DINOv3 over all frames (compiled +
    BF16) and writes the cache, then returns the mmap view.
    """
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
        ),
    )


# --------------------------------------------------------------------------
# Dataset over cached patch sequences
# --------------------------------------------------------------------------


class G1PatchSeqDataset(Dataset):
    """Yields dict with
        "patches":  (seq_len, N_patches, patch_dim)  fp16
        "proprio":  (seq_len, D_prop)                fp32
        "action":   (seq_len, action_dim)            fp32

    Cached patches stay fp16 in RAM to halve the footprint; cast to
    fp32 on the GPU per-batch in the train loop.
    """

    def __init__(
        self,
        h5_path: Path,
        patch_cache: torch.Tensor,
        *,
        seq_len: int,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.seq_len = int(seq_len)
        with h5py.File(self.h5_path, "r") as f:
            ep_offset = f["ep_offset"][:].astype(np.int64)
            ep_len = f["ep_len"][:].astype(np.int64)
            n_total = int(f["pixels"].shape[0])
            self.proprio = torch.from_numpy(f["proprio"][:]).float()
            self.action = torch.from_numpy(f["action"][:]).float()
        assert patch_cache.shape[0] == n_total, (
            f"patch cache rows {patch_cache.shape[0]} != H5 rows {n_total}"
        )
        self.patches = patch_cache
        valid: list[int] = []
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
            "patches": self.patches[s],   # fp16 (seq_len, N, 768)
            "proprio": self.proprio[s],
            "action": self.action[s],
        }


# --------------------------------------------------------------------------
# Loss (Terver recipe over patches)
# --------------------------------------------------------------------------


def compute_loss(
    model: JEPAv5,
    batch: dict,
    *,
    history_size: int,
    num_preds: int,
    rollout_weight: float,
    std_weight: float,
) -> dict:
    """Terver-style 1-step TF MSE + k-step rollout MSE, per-patch,
    plus a VICReg-style variance regularizer on the projected patch
    embeddings to prevent the trainable ``patch_proj`` from collapsing
    to a constant (the empirical failure mode of the v5 first attempt).

    batch["patches"]: (B, T, N, 768) fp16 — precomputed DINOv3 patches.
    batch["proprio"]: (B, T, D_prop)
    batch["action"]:  (B, T, action_dim)
    """
    patches_raw = batch["patches"].float()   # (B, T, N, 768)
    proprio = batch["proprio"]               # (B, T, D_prop)
    action = batch["action"]                 # (B, T, action_dim)

    # Project + proprio fuse — matches JEPAv5.encode() output.
    emb = model.patch_proj(patches_raw)      # (B, T, N, hidden)
    emb = model._fuse_proprio(emb, proprio)  # (B, T, N, hidden)

    act_emb = model.action_encoder(action)   # (B, T, hidden)

    # VICReg variance term on projected embeddings: each feature channel
    # should have std >= 1 across the (batch × time × patch) axis. If
    # patch_proj collapses to a constant, std goes to 0 → this term
    # penalizes hard.
    emb_flat = emb.flatten(0, 2)             # (B*T*N, hidden)
    emb_std = emb_flat.std(dim=0, unbiased=False) + 1e-4
    std_loss = torch.relu(1.0 - emb_std).mean()

    ctx_len = history_size
    n_preds = num_preds
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds : n_preds + ctx_len]  # (B, ctx_len, N, hidden)

    pred_emb = model.predict(ctx_emb, ctx_act)    # (B, ctx_len, N, hidden)
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
            tgt_k = emb[:, k + ctx_len - 1 : k + ctx_len]  # (B, 1, N, hidden)
            rollout_losses.append((pred_emb[:, -1:] - tgt_k).pow(2).mean())
        rollout_loss = sum(rollout_losses) / max(1, len(rollout_losses))
    else:
        rollout_loss = torch.zeros((), device=emb.device)

    total = pred_loss + rollout_weight * rollout_loss + std_weight * std_loss
    return {
        "loss": total,
        "pred_loss": pred_loss.detach(),
        "rollout_loss": rollout_loss.detach() if torch.is_tensor(rollout_loss) else torch.zeros((), device=emb.device),
        "std_loss": std_loss.detach(),
    }


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
    ap.add_argument("--dinov3-id", type=str, default=DEFAULT_DINOV3_ID)
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".vitruvian" / "m4f_v5" / "cache",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path.home() / ".vitruvian" / "m4f_v5",
    )
    ap.add_argument("--run-name", type=str, default="jepa_v5")

    # Model.
    ap.add_argument("--spatial-stride", type=int, default=2)
    ap.add_argument("--history-size", type=int, default=3)
    ap.add_argument("--num-preds", type=int, default=6)
    ap.add_argument("--predictor-hidden", type=int, default=256)
    ap.add_argument("--predictor-depth", type=int, default=6)
    ap.add_argument("--predictor-heads", type=int, default=8)
    ap.add_argument("--predictor-mlp-dim", type=int, default=1024)
    ap.add_argument("--predictor-dim-head", type=int, default=32)
    ap.add_argument("--predictor-dropout", type=float, default=0.1)
    ap.add_argument(
        "--predictor-adaln-rank",
        type=int,
        default=128,
        help="Rank of the shared AdaLN bottleneck (M4.7 refactor). "
        "128 matches the 4060 Ti memory budget; 64 trades capacity for "
        "extra VRAM if the patch predictor runs tight.",
    )
    ap.add_argument("--proprio-hidden", type=int, default=256)
    ap.add_argument("--action-frameskip", type=int, default=1)
    ap.add_argument("--action-smoothed-dim", type=int, default=10)

    # Training.
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-floor", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--rollout-weight", type=float, default=1.0)
    ap.add_argument(
        "--std-weight",
        type=float,
        default=1.0,
        help="Weight on the VICReg variance regularizer on projected "
        "patch embeddings. Set to 0 to disable (only if you're debugging "
        "collapse — normal training should keep ≥ 0.1).",
    )
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--precompute-batch",
        type=int,
        default=256,
        help="DINOv3 inference batch for the precompute pass (M4.7). "
        "Bumped from M4.6's 64 — ~3× faster precompute when combined "
        "with compile_model + bf16_autocast.",
    )
    ap.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile on the predictor. Useful for debug.",
    )
    ap.add_argument(
        "--quick-debug",
        action="store_true",
        help="1 epoch over 2000 samples for plumbing smoke test.",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"--- M4.6  JEPAv5 training (DINOv3 7×7 patches + PatchARPredictor) ---")
    print(f"device:          {device}")
    print(f"h5:              {args.h5}")
    print(f"dinov3-id:       {args.dinov3_id}")
    print(f"cache-dir:       {args.cache_dir}")
    print(f"out-dir:         {args.out_dir}")
    print(f"stride:          {args.spatial_stride}  history: {args.history_size}  num-preds: {args.num_preds}")
    print(f"batch:           {args.batch_size}  epochs: {args.epochs}  lr: {args.lr}")
    print(f"quick-debug:     {args.quick_debug}")
    print()

    # 1. Precompute patch cache — returns memory-mapped tensor.
    cache = precompute_patch_embeddings(
        args.h5,
        args.cache_dir,
        model_id=args.dinov3_id,
        spatial_stride=args.spatial_stride,
        batch_size=args.precompute_batch,
        device=device,
    )
    print(f"[cache]  {cache.describe()}")

    # 2. Dataset — dataset slices the mmap tensor; working-set RAM stays
    # O(batch size × seq_len × N × D × 2 bytes) instead of 20 GB.
    seq_len = args.history_size + args.num_preds
    dataset = G1PatchSeqDataset(args.h5, cache.tensor, seq_len=seq_len)
    print(f"[data]  samples (valid starts): {len(dataset)}  seq_len: {seq_len}")

    if args.quick_debug:
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(min(2000, len(dataset)))))
        print(f"[quick-debug] truncated to {len(dataset)} samples")
        args.epochs = 1

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

    # 3. JEPAv5.
    proprio_dim = dataset[0]["proprio"].shape[-1] if len(dataset) > 0 else 103
    action_dim = dataset[0]["action"].shape[-1] if len(dataset) > 0 else 29

    model = JEPAv5(
        dinov3_model_id=args.dinov3_id,
        spatial_stride=args.spatial_stride,
        proprio_dim=int(proprio_dim),
        proprio_hidden=args.proprio_hidden,
        action_dim=int(action_dim),
        action_frameskip=args.action_frameskip,
        action_smoothed_dim=args.action_smoothed_dim,
        predictor_num_frames=args.history_size,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        predictor_mlp_dim=args.predictor_mlp_dim,
        predictor_hidden=args.predictor_hidden,
        predictor_dim_head=args.predictor_dim_head,
        predictor_dropout=args.predictor_dropout,
        predictor_adaln_rank=args.predictor_adaln_rank,
        device=device,
        # Training reads the precomputed patch cache; no need to hold
        # the 344 MB DINOv3 weights in VRAM during training.
        backbone_lazy=True,
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen = [p for p in model.parameters() if not p.requires_grad]
    print(f"[model]  trainable: {sum(p.numel() for p in trainable):,}")
    print(f"[model]  frozen:    {sum(p.numel() for p in frozen):,}")

    # Fused AdamW — ~10% faster optimizer step on CUDA, zero accuracy change.
    opt = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, fused=True)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup = min(args.warmup_steps, total_steps // 10)

    # Compile the predictor — the expensive module. ~20-30% speedup after
    # a one-time ~30s compile stall on the first batch. backbone (DINOv3)
    # isn't invoked during training (we use cached patches) so no need to
    # compile it.
    if not args.no_compile:
        model.predictor = compile_model(model.predictor, mode="reduce-overhead")
        print(f"[compile] predictor wrapped with torch.compile")

    # 4. Train.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    step = 0
    t_start = time.perf_counter()

    jepa_config = dict(
        dinov3_model_id=args.dinov3_id,
        spatial_stride=args.spatial_stride,
        proprio_dim=int(proprio_dim),
        proprio_hidden=args.proprio_hidden,
        action_dim=int(action_dim),
        action_frameskip=args.action_frameskip,
        action_smoothed_dim=args.action_smoothed_dim,
        predictor_num_frames=args.history_size,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        predictor_mlp_dim=args.predictor_mlp_dim,
        predictor_hidden=args.predictor_hidden,
        predictor_dim_head=args.predictor_dim_head,
        predictor_dropout=args.predictor_dropout,
        predictor_adaln_rank=args.predictor_adaln_rank,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Never un-freeze DINOv3 (may be None if backbone_lazy=True).
        if model.backbone.dinov3 is not None:
            model.backbone.dinov3.eval()
        tr_pred, tr_roll, tr_std = [], [], []
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            lr_mult = cosine_lr_factor(step, total_steps, warmup, args.lr, args.lr_floor)
            for pg in opt.param_groups:
                pg["lr"] = args.lr * lr_mult

            # BF16 autocast around forward+loss. Backward dispatches mixed
            # precision automatically; no GradScaler needed (BF16 has
            # FP32's exponent range on Ampere).
            with bf16_autocast():
                info = compute_loss(
                    model, batch,
                    history_size=args.history_size,
                    num_preds=args.num_preds,
                    rollout_weight=args.rollout_weight,
                    std_weight=args.std_weight,
                )
            loss = info["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tr_pred.append(float(info["pred_loss"]))
            tr_roll.append(float(info["rollout_loss"]))
            tr_std.append(float(info["std_loss"]))
            step += 1

        model.eval()
        with torch.no_grad(), bf16_autocast():
            va_pred, va_roll = [], []
            for batch in val_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                info = compute_loss(
                    model, batch,
                    history_size=args.history_size,
                    num_preds=args.num_preds,
                    rollout_weight=args.rollout_weight,
                    std_weight=args.std_weight,
                )
                va_pred.append(float(info["pred_loss"]))
                va_roll.append(float(info["rollout_loss"]))
        tr_pred_m = sum(tr_pred) / max(1, len(tr_pred))
        tr_roll_m = sum(tr_roll) / max(1, len(tr_roll))
        va_pred_m = sum(va_pred) / max(1, len(va_pred))
        va_roll_m = sum(va_roll) / max(1, len(va_roll))
        elapsed = time.perf_counter() - t_start
        cur_lr = opt.param_groups[0]["lr"]
        tr_std_m = sum(tr_std) / max(1, len(tr_std))
        print(
            f"[epoch {epoch:>2}/{args.epochs}]  "
            f"train_pred={tr_pred_m:.4f}  train_roll={tr_roll_m:.4f}  "
            f"train_std={tr_std_m:.4f}  "
            f"val_pred={va_pred_m:.4f}  val_roll={va_roll_m:.4f}  "
            f"lr={cur_lr:.2e}  ({elapsed:.1f}s)"
        )

        # Save only trainable state (not frozen DINOv3). Strip the
        # torch.compile ``_orig_mod.`` prefix so loaders see the same
        # keys as an uncompiled model.
        def _normalize_key(k: str) -> str:
            return k.replace("predictor._orig_mod.", "predictor.")

        trainable_state = {
            _normalize_key(k): v.detach().cpu()
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

    print(f"\n=== M4.6 JEPAv5 done in {time.perf_counter() - t_start:.1f}s ===")
    print(f"best val_pred: {best_val:.4f}")
    print(f"artifacts in:  {args.out_dir}")


if __name__ == "__main__":
    main()
