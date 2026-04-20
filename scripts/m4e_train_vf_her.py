#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""M4.5 — Train a value head V_ψ on the JEPAv4 latent space with
HER-style goal relabeling.

Fork of ``m4d_train_vf.py`` with two key changes for M4.5:

1. **Backbone is JEPAv4** (frozen DINOv3 + trainable proprio/action/predictor).
   We reuse the SAME precomputed DINOv3 CLS cache as v4 training and
   apply the trained proprio encoder to fuse state.

2. **Goal sampling is HER-style** — for each transition (s_t, s_{t+1})
   in a trajectory, the goal g is a future state s_{t+k} within the
   SAME trajectory, k drawn uniformly in [1, L - t]. This gives IQL
   proper goal-reaching structure that random cross-trajectory goals
   (v1/v2) did not.

Loss / hyperparams carry over from m4d_train_vf.py v2 (LR=1e-4,
EMA=0.001, τ=0.7, γ=0.98) which converged cleanly.
"""

from __future__ import annotations

import argparse
import copy
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
sys.path.insert(0, str(ROOT / "scripts"))

from vitruvian.hwm.jepa_v4 import load_jepa_v4_from_checkpoint  # noqa: E402
from m4d_train_vf import ValueHead, ema_update  # noqa: E402
from m4e_train_jepa_v4 import precompute_embeddings  # noqa: E402


class G1HERTransitionDataset(Dataset):
    """Yields (emb_t, prop_t, emb_tp1, prop_tp1, emb_g, prop_g, is_self).

    HER: for each (t, t+1) within an episode, the goal index g is
    uniformly sampled from [t+1, episode_end]. This guarantees g is
    reachable from t within the trajectory.
    """

    def __init__(
        self,
        h5_path: Path,
        emb_cache: torch.Tensor,
        seed: int = 0,
    ) -> None:
        self.h5_path = Path(h5_path)
        with h5py.File(self.h5_path, "r") as f:
            ep_offset = f["ep_offset"][:].astype(np.int64)
            ep_len = f["ep_len"][:].astype(np.int64)
            n_total = int(f["pixels"].shape[0])
            self.proprio = torch.from_numpy(f["proprio"][:]).float()
        self.emb = emb_cache
        assert emb_cache.shape[0] == n_total

        rng = np.random.default_rng(seed)
        valid_t = []
        goal_g = []
        for o, L in zip(ep_offset, ep_len):
            o, L = int(o), int(L)
            ep_end_exclusive = o + L
            for t in range(o, o + L - 1):
                # HER-style: sample g uniformly in [t+1, ep_end_exclusive).
                g = int(rng.integers(t + 1, ep_end_exclusive))
                valid_t.append(t)
                goal_g.append(g)
        self.valid_idx = np.asarray(valid_t, dtype=np.int64)
        self.goal_idx = np.asarray(goal_g, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.valid_idx)

    def __getitem__(self, idx: int) -> dict:
        t = int(self.valid_idx[idx])
        g = int(self.goal_idx[idx])
        return {
            "emb_t": self.emb[t],
            "prop_t": self.proprio[t],
            "emb_tp1": self.emb[t + 1],
            "prop_tp1": self.proprio[t + 1],
            "emb_g": self.emb[g],
            "prop_g": self.proprio[g],
            "is_goal_self": torch.tensor(t == g, dtype=torch.bool),
        }


def fuse_emb(emb_vis: torch.Tensor, proprio: torch.Tensor, proprio_encoder: nn.Module) -> torch.Tensor:
    """emb_vis: (B, D), proprio: (B, D_prop). Returns (B, D)."""
    return emb_vis + proprio_encoder(proprio)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt-jepa-v4",
        type=Path,
        default=Path.home() / ".vitruvian" / "m4e_v4" / "best.pt",
    )
    ap.add_argument(
        "--h5",
        type=Path,
        default=Path.home() / ".stable_worldmodel" / "g1_diverse_v1.h5",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".vitruvian" / "m4e_v4" / "cache",
    )
    ap.add_argument(
        "--out-path",
        type=Path,
        default=Path.home() / ".vitruvian" / "m4e_v4" / "vf_her.pt",
    )
    ap.add_argument("--out-dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--ema-rate", type=float, default=0.001)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"--- M4.5  VF-HER training on JEPAv4 ---")
    print(f"device:         {device}")
    print(f"ckpt-jepa-v4:   {args.ckpt_jepa_v4}")
    print(f"h5:             {args.h5}")
    print(f"out-path:       {args.out_path}")
    print(f"tau: {args.tau}  gamma: {args.gamma}  ema: {args.ema_rate}")
    print(f"batch: {args.batch_size}  epochs: {args.epochs}  lr: {args.lr}")
    print()

    # 1. Precompute DINOv3 embeddings (cache hit if v4 training already ran).
    cache_path, emb_cache = precompute_embeddings(
        args.h5, args.cache_dir, device=device
    )

    # 2. Load JEPAv4 to extract its trained proprio_encoder.
    jepa = load_jepa_v4_from_checkpoint(str(args.ckpt_jepa_v4), device=device)
    proprio_encoder = jepa.proprio_encoder
    assert proprio_encoder is not None, "JEPAv4 ckpt has no proprio_encoder"
    proprio_encoder.eval()
    for p in proprio_encoder.parameters():
        p.requires_grad_(False)
    emb_dim = jepa.emb_dim
    print(f"[jepa-v4]  loaded. emb_dim={emb_dim}")

    # 3. Dataset.
    dataset = G1HERTransitionDataset(args.h5, emb_cache, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    print(f"[data]  HER pairs: {len(dataset)}  steps/epoch: {len(loader)}")

    # 4. Value head + target.
    vf = ValueHead(emb_dim=emb_dim, hidden=args.hidden, out_dim=args.out_dim).to(device)
    vf_target = copy.deepcopy(vf)
    for p in vf_target.parameters():
        p.requires_grad_(False)
    print(f"[vf]   trainable params: {sum(p.numel() for p in vf.parameters() if p.requires_grad):,}")

    opt = AdamW(vf.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    step = 0
    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        losses: list[float] = []
        tderrs: list[float] = []
        v_samples: list[torch.Tensor] = []

        for batch in loader:
            emb_t = batch["emb_t"].to(device, non_blocking=True)
            prop_t = batch["prop_t"].to(device, non_blocking=True)
            emb_tp1 = batch["emb_tp1"].to(device, non_blocking=True)
            prop_tp1 = batch["prop_tp1"].to(device, non_blocking=True)
            emb_g = batch["emb_g"].to(device, non_blocking=True)
            prop_g = batch["prop_g"].to(device, non_blocking=True)
            is_self = batch["is_goal_self"].to(device, non_blocking=True)

            with torch.no_grad():
                s_t = fuse_emb(emb_t, prop_t, proprio_encoder)
                s_tp1 = fuse_emb(emb_tp1, prop_tp1, proprio_encoder)
                s_g = fuse_emb(emb_g, prop_g, proprio_encoder)

            V_st = vf(s_t, s_g)
            with torch.no_grad():
                V_snext = vf_target(s_tp1, s_g)

            r = torch.full_like(V_st, -1.0)
            r = torch.where(is_self, torch.zeros_like(r), r)

            td_error = r + args.gamma * V_snext - V_st
            weight = torch.where(
                td_error < 0,
                torch.full_like(td_error, 1.0 - args.tau),
                torch.full_like(td_error, args.tau),
            )
            loss = (weight * td_error.pow(2)).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ema_update(vf_target, vf, args.ema_rate)

            losses.append(float(loss.detach()))
            tderrs.append(float(td_error.detach().abs().mean()))
            v_samples.append(V_st.detach())
            step += 1

        mean_loss = sum(losses) / max(1, len(losses))
        mean_tderr = sum(tderrs) / max(1, len(tderrs))
        V_all = torch.cat(v_samples, dim=0)
        V_mean = float(V_all.mean())
        V_min = float(V_all.min())
        V_max = float(V_all.max())
        elapsed = time.perf_counter() - t_start
        print(
            f"[epoch {epoch:>2}/{args.epochs}]  loss={mean_loss:.4f}  "
            f"|td|={mean_tderr:.4f}  V∈[{V_min:.2f},{V_max:.2f}]  "
            f"mean={V_mean:.2f}  ({elapsed:.1f}s)"
        )

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(
                {
                    "vf_state": vf.state_dict(),
                    "out_dim": args.out_dim,
                    "hidden": args.hidden,
                    "emb_dim": emb_dim,
                    "epoch": epoch,
                    "loss": mean_loss,
                    "tau": args.tau,
                    "gamma": args.gamma,
                    "ema_rate": args.ema_rate,
                    "her": True,
                },
                args.out_path,
            )

    print(f"\n=== M4.5 VF-HER done in {time.perf_counter() - t_start:.1f}s ===")
    print(f"best loss: {best_loss:.4f}")
    print(f"saved: {args.out_path}")


if __name__ == "__main__":
    main()
