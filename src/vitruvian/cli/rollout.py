# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""``vit-rollout`` — open-loop latent rollout accuracy (thesis question Q1a).

The world model's *true* quality metric: given the first ``history_size``
frames of a held-out episode, roll the latent forward open-loop under the
episode's **recorded** actions and compare the predicted embeddings to the
ground-truth encoded embeddings, as a function of horizon.

Both live in the predictor's *projected* latent — the exact space the loss
trains against (:func:`vitruvian.training.losses.prediction_loss`) and the
space the MPPI cost is computed in — so the number is directly meaningful.

We report the model against a **persistence baseline** (predict "no change":
the last history frame, held constant). If the model does not beat
persistence — especially as the horizon grows and one-step errors compound —
the world model is not capturing dynamics, and the planner that rolls it out
is decorative. That is the crux of Q1 ("does the planner earn its keep?").

Runs on the frozen M5 baseline; no training. Reuses the same DINOv3 patch
cache ``vit-train`` built (mmap load, no recompute) and the same
episode-aware held-out split, so the eval episodes are exactly the ones
training never saw.

Usage::

    uv run vit-rollout configs/eval/rollout_accuracy.yaml \
        --ckpt ~/.vitruvian/jepa_v5/best.pt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from vitruvian.data import (
    CacheKey,
    EmbeddingCache,
    G1PatchSeqDataset,
    episode_aware_split,
)
from vitruvian.models import JEPA, load_jepa
from vitruvian.utils import load_config


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return float((a @ b) / (a.norm() * b.norm() + 1e-9))


def rollout_accuracy(
    jepa: JEPA,
    patch_cache: torch.Tensor,
    ep_offset: np.ndarray,
    ep_len: np.ndarray,
    action: torch.Tensor,
    episodes: Sequence[int],
    *,
    history_size: int,
    max_horizon: int,
    device: str,
) -> dict[str, Any]:
    """Open-loop rollout accuracy over ``episodes``.

    For each episode, encode (project) the first ``history_size`` frames as
    the rollout seed, roll forward under the recorded actions, and score the
    predicted vs ground-truth projected embeddings at each horizon with cosine
    similarity and MSE — alongside a persistence ("no-change") baseline.

    Args:
        jepa: a patch JEPA (must carry ``patch_projector``).
        patch_cache: ``(N_total, N_patch, D_raw)`` precomputed backbone
            embeddings (the ``vit-train`` cache tensor).
        ep_offset / ep_len: per-episode start row and length.
        action: ``(N_total, action_dim)`` recorded actions.
        episodes: episode indices to evaluate (the held-out set).
        history_size: number of seed frames ``H``.
        max_horizon: max steps-ahead to score (capped per-episode by length).
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        Dict with ``per_horizon`` (model vs persistence cos/mse per horizon),
        ``n_episodes``, and a ``summary``.
    """
    jepa.eval()
    proj = jepa.patch_projector
    if proj is None:
        raise ValueError(
            "rollout_accuracy requires a patch JEPA carrying a patch_projector"
        )
    H = int(history_size)

    m_cos: dict[int, list[float]] = defaultdict(list)
    m_mse: dict[int, list[float]] = defaultdict(list)
    p_cos: dict[int, list[float]] = defaultdict(list)
    p_mse: dict[int, list[float]] = defaultdict(list)
    n_used = 0

    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), autocast:
        for e in episodes:
            off, L = int(ep_offset[e]), int(ep_len[e])
            K = min(int(max_horizon), L - H)
            if K < 1:
                continue
            n_used += 1

            gt_raw = patch_cache[off : off + H + K].float().to(device)
            gt_proj = proj(gt_raw)  # (H+K, N, hidden)
            emb_init = gt_proj[:H].unsqueeze(0).unsqueeze(0)  # (1,1,H,N,hidden)
            acts = (
                action[off : off + H + K].to(device).unsqueeze(0).unsqueeze(0)
            )  # (1, 1, H+K, action_dim)

            out = jepa.rollout({"emb": emb_init}, acts, history_size=H)
            pred = out["predicted_emb"][0, 0]  # (H+K+1, N, hidden)

            last_hist = gt_proj[H - 1].float()  # persistence prediction
            for h in range(1, K + 1):
                idx = H + h - 1  # predicted / ground-truth frame in the slice
                pr = pred[idx].float()
                gt = gt_proj[idx].float()
                m_cos[h].append(_cosine(pr, gt))
                m_mse[h].append(float((pr - gt).pow(2).mean()))
                p_cos[h].append(_cosine(last_hist, gt))
                p_mse[h].append(float((last_hist - gt).pow(2).mean()))

    per_horizon: list[dict[str, Any]] = []
    for h in range(1, int(max_horizon) + 1):
        if not m_cos[h]:
            continue
        per_horizon.append(
            {
                "horizon": h,
                "n": len(m_cos[h]),
                "model_cos": float(np.mean(m_cos[h])),
                "model_mse": float(np.mean(m_mse[h])),
                "persist_cos": float(np.mean(p_cos[h])),
                "persist_mse": float(np.mean(p_mse[h])),
            }
        )

    mean_model = float(np.mean([r["model_cos"] for r in per_horizon])) if per_horizon else float("nan")
    mean_persist = float(np.mean([r["persist_cos"] for r in per_horizon])) if per_horizon else float("nan")
    return {
        "history_size": H,
        "max_horizon": int(max_horizon),
        "n_episodes": n_used,
        "per_horizon": per_horizon,
        "summary": {
            "mean_model_cos": mean_model,
            "mean_persist_cos": mean_persist,
            "beats_persistence": bool(mean_model > mean_persist),
            "final_horizon_model_cos": per_horizon[-1]["model_cos"] if per_horizon else float("nan"),
            "final_horizon_persist_cos": per_horizon[-1]["persist_cos"] if per_horizon else float("nan"),
        },
    }


def _held_out_episodes(
    dataset: G1PatchSeqDataset, *, val_frac: float, seed: int, seq_len: int
) -> list[int]:
    """The episodes assigned to validation by the training split — the ones
    the world model never saw."""
    _, val_idx = episode_aware_split(
        dataset.ep_offset,
        dataset.valid_idx,
        val_frac=val_frac,
        seed=seed,
        seq_len=seq_len,
    )
    starts = dataset.valid_idx[val_idx]
    ep_id = np.searchsorted(dataset.ep_offset, starts, side="right") - 1
    return sorted({int(e) for e in ep_id})


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Open-loop latent rollout accuracy (Q1a)."
    )
    ap.add_argument("config", type=Path, help="YAML config path")
    ap.add_argument("--override", "-o", action="append", default=[])
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Cap the number of held-out episodes evaluated (debug/speed).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = args.ckpt or Path(cfg["ckpt"]).expanduser()
    jepa = load_jepa(ckpt, device=device)
    # We score against the precomputed cache + patch_projector, so the DINOv3
    # backbone forward is never called — leave it lazy (saves VRAM).

    data_cfg = cfg["data"]
    h5_path = Path(data_cfg["h5"]).expanduser()
    key = CacheKey(
        h5_path=h5_path,
        model_id=data_cfg.get(
            "model_id", "facebook/dinov3-vitb16-pretrain-lvd1689m"
        ),
        mode=data_cfg.get("cache_mode", "patch2"),
    )
    cache = EmbeddingCache.load(
        cache_dir=Path(data_cfg["cache_dir"]).expanduser(), key=key
    )
    if cache is None:
        raise SystemExit(
            f"cache MISS for {key.filename()} at {data_cfg['cache_dir']}; "
            f"run vit-train on {h5_path.name} first to build it."
        )

    history_size = int(cfg.get("history_size", 3))
    train_seq_len = int(cfg.get("train_seq_len", 9))
    dataset = G1PatchSeqDataset(h5_path, cache.tensor, seq_len=train_seq_len)
    episodes = _held_out_episodes(
        dataset,
        val_frac=float(cfg.get("val_frac", 0.05)),
        seed=int(cfg.get("seed", 0)),
        seq_len=train_seq_len,
    )
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    res = rollout_accuracy(
        jepa,
        dataset.patches,
        dataset.ep_offset,
        dataset.ep_len,
        dataset.action,
        episodes,
        history_size=history_size,
        max_horizon=int(cfg.get("max_horizon", 16)),
        device=device,
    )

    print(f"[rollout-eval] held-out episodes: {res['n_episodes']}")
    print("horizon  model_cos  persist_cos  model_mse  persist_mse    n")
    for r in res["per_horizon"]:
        print(
            f"{r['horizon']:>5}   {r['model_cos']:>8.4f}   "
            f"{r['persist_cos']:>9.4f}   {r['model_mse']:>8.4f}   "
            f"{r['persist_mse']:>9.4f}  {r['n']:>4}"
        )
    s = res["summary"]
    verdict = "BEATS" if s["beats_persistence"] else "does NOT beat"
    print(
        f"[rollout-eval] mean model_cos={s['mean_model_cos']:.4f} vs "
        f"persist_cos={s['mean_persist_cos']:.4f} -> world model {verdict} "
        f"persistence"
    )

    out_dir = (
        Path(args.out_dir).expanduser()
        if args.out_dir is not None
        else Path(cfg.get("out_dir", "/tmp/vitruvian/rollout_eval")).expanduser()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rollout_accuracy.json").write_text(json.dumps(res, indent=2))
    with (out_dir / "rollout_accuracy.tsv").open("w") as tf:
        tf.write("horizon\tmodel_cos\tpersist_cos\tmodel_mse\tpersist_mse\tn\n")
        for r in res["per_horizon"]:
            tf.write(
                f"{r['horizon']}\t{r['model_cos']:.6f}\t"
                f"{r['persist_cos']:.6f}\t{r['model_mse']:.6f}\t"
                f"{r['persist_mse']:.6f}\t{r['n']}\n"
            )
    print(f"[rollout-eval] wrote {out_dir / 'rollout_accuracy.json'}")


if __name__ == "__main__":
    main()
