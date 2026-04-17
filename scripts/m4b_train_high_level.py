#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""M4.4b — train the HWM-on-LeWM high-level (macro-action) world model.

Loads a frozen LeWM JEPA encoder (from M4.3's Lightning ckpt), plugs it
into ``HighLevelModel`` (MacroActionEncoder + causal-transformer
predictor), and trains the predictor to forecast next-macro CLS
embeddings under teacher-forced macro-action rollouts.

Runs in the main vitruvian venv (Python 3.12). The script adds
``external/le-wm/`` to ``sys.path`` so that ``jepa.JEPA`` and
``module.Embedder`` can be imported without installing
stable_pretraining.

Usage
-----
    uv run python scripts/m4b_train_high_level.py --quick-debug
    uv run python scripts/m4b_train_high_level.py --epochs 10 --wandb

See docs/decisions/008-hwm-planning-layer.md for the architectural
rationale (flat CLS adapter, MLP predictor, step_skip=50).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "le-wm"))

from vitruvian.hwm.backbone_adapter import (  # noqa: E402
    LeWMBackboneAdapter,
    load_lewm_jepa_from_checkpoint,
)
from vitruvian.hwm.data import G1WaypointDataset  # noqa: E402
from vitruvian.hwm.high_level import HighLevelModel  # noqa: E402
from vitruvian.hwm.objectives import PredictionLoss, VICRegLoss  # noqa: E402


STABLEWM_HOME = Path.home() / ".stable_worldmodel"
DEFAULT_CKPT = STABLEWM_HOME / "lewm_g1_seed_weights.ckpt"
DEFAULT_H5 = STABLEWM_HOME / "g1_joystick_expert.h5"
DEFAULT_OUT = Path.home() / ".vitruvian" / "m4b_hl_v1"


def cosine_lr(step: int, total: int, warmup: int, peak: float, floor: float) -> float:
    """Linear warmup then cosine decay from peak→floor, returned as a
    multiplier against AdamW's base lr (we set base_lr = peak and
    return in [floor/peak, 1])."""
    if step < warmup:
        return float(step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    lr = floor + (peak - floor) * cos
    return lr / peak


def build_model(
    ckpt_lewm: Path,
    *,
    step_skip: int,
    macro_act_dim: int,
    hl_hidden: int,
    hl_layers: int,
    hl_heads: int,
    n_macros: int,
    device: str,
) -> HighLevelModel:
    print(f"[lewm] loading frozen JEPA from {ckpt_lewm}")
    t0 = time.time()
    jepa = load_lewm_jepa_from_checkpoint(
        str(ckpt_lewm),
        lewm_repo_path=str(ROOT / "external" / "le-wm"),
        device=device,
    )
    print(f"[lewm]   loaded in {time.time() - t0:.1f}s")

    # max_seq_len = n_macros (we feed N macros, predict N nexts).
    model = HighLevelModel(
        lewm_jepa=jepa,
        action_dim=29,
        step_skip=step_skip,
        macro_act_dim=macro_act_dim,
        hidden=hl_hidden,
        n_layers=hl_layers,
        n_heads=hl_heads,
        max_seq_len=max(n_macros, 4),
        freeze_backbone=True,
    ).to(device)

    # Assert backbone frozen per plan Stage-1 blocker.
    backbone_trainable = sum(
        p.numel() for p in model.backbone.parameters() if p.requires_grad
    )
    assert backbone_trainable == 0, (
        f"Backbone is not frozen: {backbone_trainable} trainable params leak."
    )
    total_trainable = model.n_trainable()
    print(
        f"[model] backbone trainable: {backbone_trainable}  "
        f"(expected 0)   HL trainable: {total_trainable:,}"
    )
    return model


def train_epoch(
    model: HighLevelModel,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    pred_loss_fn: PredictionLoss,
    vicreg_fn: VICRegLoss,
    device: str,
    epoch: int,
    use_wandb: bool,
    log_every: int = 20,
) -> dict:
    model.train()
    # Keep backbone in eval (frozen; dropout/BN disabled).
    model.backbone.eval()

    agg = {
        "pred": 0.0,
        "std": 0.0,
        "cov": 0.0,
        "total": 0.0,
    }
    n = 0

    for step, batch in enumerate(loader):
        pixels = batch["pixels"].to(device, non_blocking=True)  # (B, N+1, 3, H, W)
        macro_actions = batch["macro_actions"].to(
            device, non_blocking=True
        )  # (B, N, step_skip, 29)

        with torch.no_grad():
            emb = model.encode(pixels)  # (B, N+1, D)

        preds = model.predict_next_latents(emb, macro_actions)  # (B, N, D)
        target = emb[:, 1:]

        pred_loss = pred_loss_fn(preds, target)
        vicreg = vicreg_fn(preds)
        total = pred_loss + vicreg["total"]

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        bs = pixels.shape[0]
        n += bs
        agg["pred"] += float(pred_loss.detach()) * bs
        agg["std"] += float(vicreg["std_loss"].detach()) * bs
        agg["cov"] += float(vicreg["cov_loss"].detach()) * bs
        agg["total"] += float(total.detach()) * bs

        if step % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  e{epoch} s{step:>4}  "
                f"pred={pred_loss.item():.4f}  "
                f"std={vicreg['std_loss'].item():.4f}  "
                f"cov={vicreg['cov_loss'].item():.4f}  "
                f"total={total.item():.4f}  lr={lr:.2e}"
            )
            if use_wandb:
                import wandb

                wandb.log(
                    {
                        "train/pred_loss": float(pred_loss),
                        "train/std_loss": float(vicreg["std_loss"]),
                        "train/cov_loss": float(vicreg["cov_loss"]),
                        "train/total_loss": float(total),
                        "train/lr": lr,
                        "train/epoch": epoch,
                    }
                )

    return {k: v / max(1, n) for k, v in agg.items()}


@torch.no_grad()
def validate(
    model: HighLevelModel,
    loader: DataLoader,
    pred_loss_fn: PredictionLoss,
    vicreg_fn: VICRegLoss,
    device: str,
) -> dict:
    model.eval()
    agg = {"pred": 0.0, "std": 0.0, "cov": 0.0, "total": 0.0, "cos": 0.0}
    n = 0
    for batch in loader:
        pixels = batch["pixels"].to(device, non_blocking=True)
        macro_actions = batch["macro_actions"].to(device, non_blocking=True)
        emb = model.encode(pixels)
        preds = model.predict_next_latents(emb, macro_actions)
        target = emb[:, 1:]
        pl = pred_loss_fn(preds, target)
        vr = vicreg_fn(preds)
        cos = torch.nn.functional.cosine_similarity(
            preds.flatten(0, 1), target.flatten(0, 1), dim=-1
        ).mean()
        bs = pixels.shape[0]
        n += bs
        agg["pred"] += float(pl) * bs
        agg["std"] += float(vr["std_loss"]) * bs
        agg["cov"] += float(vr["cov_loss"]) * bs
        agg["total"] += float(pl + vr["total"]) * bs
        agg["cos"] += float(cos) * bs
    return {k: v / max(1, n) for k, v in agg.items()}


def save_checkpoint(
    model: HighLevelModel,
    optimizer: AdamW,
    out_dir: Path,
    epoch: int,
    metrics: dict,
    config: dict,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"epoch_{epoch:03d}.pt"
    # Only save the TRAINABLE parts (action_encoder + predictor); the
    # frozen LeWM backbone is reloaded from its own ckpt at inference.
    payload = {
        "epoch": epoch,
        "action_encoder": model.action_encoder.state_dict(),
        "predictor": model.predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }
    torch.save(payload, ckpt_path)
    latest = out_dir / "latest.pt"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(ckpt_path.name)
    size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    print(f"[ckpt] -> {ckpt_path.name} ({size_mb:.1f} MB), latest.pt updated")
    return ckpt_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-lewm", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--h5", type=Path, default=DEFAULT_H5)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)

    ap.add_argument("--step-skip", type=int, default=50)
    # n-macros=3 matches the paper's Franka setup (N=3). Our expert
    # episodes cap at 200 frames (4s at 50Hz), so larger n-macros + any
    # future goal offset overflows. Training doesn't consume goal_pixel,
    # so goal-offset defaults to 0.
    ap.add_argument("--n-macros", type=int, default=3)
    ap.add_argument("--goal-offset-lo", type=int, default=0)
    ap.add_argument("--goal-offset-hi", type=int, default=0)

    ap.add_argument("--macro-act-dim", type=int, default=32)
    ap.add_argument("--hl-hidden", type=int, default=256)
    ap.add_argument("--hl-layers", type=int, default=4)
    ap.add_argument("--hl-heads", type=int, default=4)

    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-floor", type=float, default=3e-6)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--vicreg-std", type=float, default=25.0)
    ap.add_argument("--vicreg-cov", type=float, default=1.0)

    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--run-name", type=str, default="m4b-hl-v1")
    ap.add_argument(
        "--quick-debug",
        action="store_true",
        help="1 epoch over 200 samples, num_workers=0. "
        "Stage-1 plumbing smoke test.",
    )
    args = ap.parse_args()

    if args.quick_debug:
        args.epochs = 1
        args.num_workers = 0
        args.batch_size = min(args.batch_size, 8)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- M4.4b high-level training — {args.run_name} ---")
    print(f"device:          {device}")
    print(f"ckpt-lewm:       {args.ckpt_lewm}")
    print(f"h5:              {args.h5}")
    print(f"step-skip:       {args.step_skip}  n-macros: {args.n_macros}")
    print(f"batch-size:      {args.batch_size}  epochs: {args.epochs}")
    print(f"lr:              {args.lr}  (cos decay to {args.lr_floor})")
    print(f"quick-debug:     {args.quick_debug}")
    print()

    if not args.ckpt_lewm.exists():
        raise FileNotFoundError(f"LeWM checkpoint not found: {args.ckpt_lewm}")
    if not args.h5.exists():
        raise FileNotFoundError(f"Expert H5 not found: {args.h5}")

    # Dataset + split
    full = G1WaypointDataset(
        h5_path=str(args.h5),
        step_skip=args.step_skip,
        n_macros=args.n_macros,
        goal_offset_range=(args.goal_offset_lo, args.goal_offset_hi),
        seed=args.seed,
    )
    n_total = len(full)
    n_val = max(1, int(n_total * args.val_frac))
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(args.seed))
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()

    if args.quick_debug:
        train_idx = train_idx[:200]
        val_idx = val_idx[:50]

    train_ds = Subset(full, train_idx)
    val_ds = Subset(full, val_idx)
    print(f"[data] samples: {n_total}  train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    # Model
    model = build_model(
        args.ckpt_lewm,
        step_skip=args.step_skip,
        macro_act_dim=args.macro_act_dim,
        hl_hidden=args.hl_hidden,
        hl_layers=args.hl_layers,
        hl_heads=args.hl_heads,
        n_macros=args.n_macros,
        device=device,
    )

    # Objectives
    pred_loss_fn = PredictionLoss(coeff=1.0)
    vicreg_fn = VICRegLoss(std_coeff=args.vicreg_std, cov_coeff=args.vicreg_cov)

    # Optimizer + schedule
    optimizer = AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_lr(
            s, total_steps, args.warmup_steps, args.lr, args.lr_floor
        ),
    )

    # wandb
    if args.wandb:
        import wandb

        wandb.init(
            project="vitruvian",
            name=args.run_name,
            tags=["m4.4b", "hwm", "lewm", "g1"],
            config={
                **vars(args),
                "hl_trainable_params": model.n_trainable(),
            },
        )

    cfg = {
        "step_skip": args.step_skip,
        "n_macros": args.n_macros,
        "macro_act_dim": args.macro_act_dim,
        "hl_hidden": args.hl_hidden,
        "hl_layers": args.hl_layers,
        "hl_heads": args.hl_heads,
        "backbone_ckpt": str(args.ckpt_lewm),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            pred_loss_fn,
            vicreg_fn,
            device,
            epoch,
            use_wandb=args.wandb,
        )
        val = validate(model, val_loader, pred_loss_fn, vicreg_fn, device)
        dt = time.time() - t0
        print(
            f"[epoch {epoch:>2}/{args.epochs}] "
            f"train_pred={tr['pred']:.4f} val_pred={val['pred']:.4f} "
            f"val_cos={val['cos']:+.3f}  val_std={val['std']:.3f}  "
            f"val_cov={val['cov']:.3f}  ({dt:.1f}s)"
        )
        if args.wandb:
            import wandb

            wandb.log(
                {
                    "epoch": epoch,
                    "validate/pred_loss": val["pred"],
                    "validate/std_loss": val["std"],
                    "validate/cov_loss": val["cov"],
                    "validate/total_loss": val["total"],
                    "validate/cosine": val["cos"],
                    "train/pred_loss_epoch": tr["pred"],
                    "train/total_loss_epoch": tr["total"],
                    "train/epoch_wall_s": dt,
                }
            )
        save_checkpoint(
            model, optimizer, args.out_dir, epoch, {"train": tr, "val": val}, cfg
        )
        if val["pred"] < best_val:
            best_val = val["pred"]
            best = args.out_dir / "best.pt"
            if best.is_symlink() or best.exists():
                best.unlink()
            best.symlink_to(f"epoch_{epoch:03d}.pt")

    total_wall = time.time() - start
    print(f"\n=== M4.4b done in {total_wall:.1f}s ({total_wall / 60:.1f} min) ===")
    print(f"best val_pred: {best_val:.4f}")
    print(f"artifacts in:  {args.out_dir}")
    if args.wandb:
        import wandb

        wandb.summary["wall_seconds_total"] = total_wall
        wandb.summary["best_val_pred_loss"] = best_val
        wandb.finish()


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
