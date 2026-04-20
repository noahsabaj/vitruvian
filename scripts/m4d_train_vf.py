#!/usr/bin/env python
"""M4.4d — Train a value head V_ψ on frozen LeWM v3 encoder (VF_quasi).

Following Destrade et al. (arXiv:2601.00844), learn an MLP f_ψ: emb_192 →
R^d such that V(s, g) = -||f_ψ(s) - f_ψ(g)||² approximates the negative
goal-conditioned cost-to-go. Trained via Implicit Q-Learning expectile
regression on offline G1 expert trajectories. At plan time the MPPI
cost is replaced with ||f_ψ(pred_final) - f_ψ(goal)||², so the cost
surface is shaped by the learned value rather than terminal-MSE over
imprecise 50-step rollouts.

Pixel-only (no proprio fusion) so the embedding pipeline matches what
the current flat MPPI planner consumes.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "le-wm"))

from vitruvian.hwm.backbone_adapter import (  # noqa: E402
    LeWMBackboneAdapter,
    load_lewm_jepa_from_checkpoint,
)


class G1TransitionDataset(Dataset):
    """Yields (pixel_t, pixel_{t+1}, pixel_g) triples from the G1 expert
    HDF5. Goals are sampled on-the-fly in __getitem__: 50/50 mix of
    (a) trajectory endpoints and (b) uniformly random frames. Entire
    pixel array (~2.9 GB, 19,690 × 224 × 224 × 3 uint8) is cached in
    host RAM at __init__ because random access against the HDF5 file
    is orders of magnitude slower than RAM indexing (empirically the
    file-backed path deadlocks DataLoader for >1h)."""

    def __init__(self, h5_path: Path, seed: int = 0) -> None:
        self.h5_path = Path(h5_path)
        with h5py.File(self.h5_path, "r") as f:
            ep_offset = f["ep_offset"][:].astype(np.int64)
            ep_len = f["ep_len"][:].astype(np.int64)
            # Materialize ALL pixels into RAM. Bigger RAM hit but random
            # access becomes O(1).
            self.pixels = f["pixels"][:]
            self.n_total = int(self.pixels.shape[0])
        self.ep_offset = ep_offset
        self.ep_len = ep_len
        self.ep_end = ep_offset + ep_len

        # Valid (t, t+1) pairs: any t with t+1 in the same episode.
        valid = []
        for o, L in zip(ep_offset, ep_len):
            for t in range(int(o), int(o + L - 1)):
                valid.append(t)
        self.valid_idx = np.asarray(valid, dtype=np.int64)

        # Per-sample random goal: 50% endpoint, 50% uniform.
        rng = np.random.default_rng(seed)
        self._goal_random = rng.integers(0, self.n_total, size=len(self.valid_idx))
        which_ep = np.searchsorted(ep_offset, self.valid_idx, side="right") - 1
        endpoint_for_each = self.ep_end[which_ep] - 1
        use_endpoint = rng.random(len(self.valid_idx)) < 0.5
        self._goal_idx = np.where(use_endpoint, endpoint_for_each, self._goal_random)

    def __len__(self) -> int:
        return len(self.valid_idx)

    def __getitem__(self, idx: int) -> dict:
        t = int(self.valid_idx[idx])
        g = int(self._goal_idx[idx])
        # (H, W, 3) uint8 -> (3, H, W). Copy is needed because pytorch
        # share-memory on numpy slices can fight the DataLoader.
        px_t = torch.from_numpy(self.pixels[t]).permute(2, 0, 1).contiguous()
        px_tp1 = torch.from_numpy(self.pixels[t + 1]).permute(2, 0, 1).contiguous()
        px_g = torch.from_numpy(self.pixels[g]).permute(2, 0, 1).contiguous()
        return {
            "px_t": px_t,
            "px_tp1": px_tp1,
            "px_g": px_g,
            "is_goal_self": torch.tensor(t == g, dtype=torch.bool),
        }


class ValueHead(nn.Module):
    """f_ψ: emb_192 → R^{out_dim}. V(s, g) = -||f(s) - f(g)||² ≈ V*(s, g)."""

    def __init__(self, emb_dim: int = 192, hidden: int = 256, out_dim: int = 128) -> None:
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, s_emb: torch.Tensor, g_emb: torch.Tensor) -> torch.Tensor:
        # Returns scalar V ≤ 0.
        return -((self.f(s_emb) - self.f(g_emb)) ** 2).sum(-1)


def ema_update(target: nn.Module, source: nn.Module, rate: float) -> None:
    with torch.no_grad():
        for p_t, p_s in zip(target.parameters(), source.parameters()):
            p_t.data.mul_(1.0 - rate).add_(p_s.data, alpha=rate)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt-lewm",
        type=Path,
        default=Path.home() / ".stable_worldmodel" / "lewm_g1_v3_weights.ckpt",
    )
    ap.add_argument(
        "--h5",
        type=Path,
        default=Path.home() / ".stable_worldmodel" / "g1_joystick_expert.h5",
    )
    ap.add_argument(
        "--out-path",
        type=Path,
        default=Path.home() / ".vitruvian" / "m4d_vf_v1" / "vf.pt",
    )
    ap.add_argument("--out-dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--tau", type=float, default=0.8, help="IQL expectile")
    ap.add_argument("--gamma", type=float, default=0.98, help="Discount")
    ap.add_argument("--ema-rate", type=float, default=0.005)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"--- M4.4d  value-head training (VF_quasi) ---")
    print(f"device:     {device}")
    print(f"ckpt-lewm:  {args.ckpt_lewm}")
    print(f"h5:         {args.h5}")
    print(f"out-path:   {args.out_path}")
    print(f"tau:        {args.tau}  gamma: {args.gamma}")
    print(f"batch-size: {args.batch_size}  epochs: {args.epochs}  lr: {args.lr}")
    print()

    # 1. Frozen v3 encoder
    jepa = load_lewm_jepa_from_checkpoint(
        str(args.ckpt_lewm), lewm_repo_path=str(ROOT / "external" / "le-wm"), device=device
    )
    backbone = LeWMBackboneAdapter(jepa, freeze=True).to(device)
    emb_dim = backbone.output_dim
    print(f"[lewm] frozen encoder output dim: {emb_dim}")

    # 2. Dataset
    dataset = G1TransitionDataset(args.h5, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    print(f"[data] valid (t, t+1) pairs: {len(dataset)}")
    print(f"[data] steps / epoch:        {len(loader)}")

    # 3. Value head + target
    vf = ValueHead(emb_dim=emb_dim, hidden=args.hidden, out_dim=args.out_dim).to(device)
    vf_target = copy.deepcopy(vf)
    for p in vf_target.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in vf.parameters() if p.requires_grad)
    print(f"[vf]   trainable params: {n_params:,}")

    opt = AdamW(vf.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 4. Train
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    step = 0
    best_loss = float("inf")
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        ep_losses: list[float] = []
        ep_tderrs: list[float] = []
        for batch in loader:
            px_t = batch["px_t"].to(device, non_blocking=True)
            px_tp1 = batch["px_tp1"].to(device, non_blocking=True)
            px_g = batch["px_g"].to(device, non_blocking=True)
            # (B, T=1, 3, H, W) for adapter
            with torch.no_grad():
                emb_t = backbone.encode(px_t.unsqueeze(1)).squeeze(1)    # (B, D)
                emb_tp1 = backbone.encode(px_tp1.unsqueeze(1)).squeeze(1)
                emb_g = backbone.encode(px_g.unsqueeze(1)).squeeze(1)

            V_st = vf(emb_t, emb_g)  # (B,)
            with torch.no_grad():
                V_snext = vf_target(emb_tp1, emb_g)

            # Reward: -1 unless s == g. In continuous pixel space that's
            # essentially always -1; we still special-case the exact
            # self-goal to avoid biasing the expectile.
            r = torch.full_like(V_st, -1.0)
            r = torch.where(batch["is_goal_self"].to(device), torch.zeros_like(r), r)

            td_error = r + args.gamma * V_snext - V_st  # (B,)
            # Expectile weight: larger for positive TD errors (τ>0.5),
            # pushing V upward — standard IQL conservative estimate.
            weight = torch.where(
                td_error < 0,
                torch.full_like(td_error, 1.0 - args.tau),
                torch.full_like(td_error, args.tau),
            )
            loss = (weight * td_error.pow(2)).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            ema_update(vf_target, vf, args.ema_rate)

            ep_losses.append(float(loss.detach()))
            ep_tderrs.append(float(td_error.detach().abs().mean()))
            step += 1

        mean_loss = sum(ep_losses) / max(1, len(ep_losses))
        mean_tderr = sum(ep_tderrs) / max(1, len(ep_tderrs))
        # Report V stats on the last batch.
        with torch.no_grad():
            V_mean = float(V_st.mean())
            V_min = float(V_st.min())
            V_max = float(V_st.max())
        elapsed = time.perf_counter() - t_start
        print(
            f"[epoch {epoch:>2}/{args.epochs}] loss={mean_loss:.4f}  |td|={mean_tderr:.4f}  "
            f"V∈[{V_min:.2f},{V_max:.2f}] mean={V_mean:.2f}  "
            f"({elapsed:.1f}s)"
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
                },
                args.out_path,
            )

    print(f"\n=== M4.4d done in {time.perf_counter() - t_start:.1f}s ===")
    print(f"best loss: {best_loss:.4f}")
    print(f"saved: {args.out_path}")


if __name__ == "__main__":
    main()
