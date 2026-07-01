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


def action_sensitivity(
    jepa: JEPA,
    patch_cache: torch.Tensor,
    ep_offset: np.ndarray,
    ep_len: np.ndarray,
    action: torch.Tensor,
    episodes: Sequence[int],
    *,
    history_size: int,
    horizon: int,
    device: str,
    n_cand: int = 64,
    noise_sigma: float = 0.3,
    seed: int = 0,
) -> dict[str, Any]:
    """How much do *different* action sequences change the predicted terminal
    latent? If "barely" relative to the forward motion, MPPI has nothing to
    discriminate on — the mechanistic cause of weak steering (Delta-JEPA /
    2606.30068 hypothesis).

    Per held-out episode: seed with H history frames, roll out ``n_cand``
    candidate action sequences, and measure the spread of the predicted terminal
    latents two ways — "local" (recorded actions + N(0, noise_sigma), MPPI's own
    regime) and "diverse" (uniform in the action box, an upper bound on
    responsiveness). Reports ``spread`` (RMS distance of terminals from their
    centroid), ``fwd`` (mean distance from the seed latent), and
    ``spread/fwd`` (≈ fraction of forward motion that is action-controllable),
    plus ``cost_cv`` (coefficient of variation of MSE-to-an-in-episode-goal
    across the local candidates — the exact quantity MPPI ranks on; ~0 ⇒ MPPI
    cannot discriminate).
    """
    jepa.eval()
    proj = jepa.patch_projector
    if proj is None:
        raise ValueError("action_sensitivity requires a patch JEPA (patch_projector)")
    H, A = int(history_size), int(action.shape[-1])
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    keys = (
        "local_spread", "local_fwd", "local_ratio",
        "diverse_spread", "diverse_fwd", "diverse_ratio", "cost_cv", "traj_std",
    )
    acc: dict[str, list[float]] = {k: [] for k in keys}
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), autocast:
        for e in episodes:
            off, L = int(ep_offset[e]), int(ep_len[e])
            if L < H + horizon + 1:
                continue
            gt = proj(patch_cache[off : off + L].float().to(device))  # (L, N, D)
            emb_init = gt[:H]
            z_init = emb_init[-1].reshape(-1)
            z_goal = gt[H + horizon].reshape(-1)  # in-episode goal
            acc["traj_std"].append(float(gt.reshape(L, -1).std(dim=0).mean()))
            acts_rec = action[off : off + H + horizon].to(device)  # (H+horizon, A)

            cand_sets = {
                "local": (
                    acts_rec.unsqueeze(0)
                    + noise_sigma
                    * torch.randn(n_cand, H + horizon, A, generator=gen).to(device)
                ).clamp(-1.0, 1.0),
                "diverse": (
                    2.0 * torch.rand(n_cand, H + horizon, A, generator=gen).to(device)
                    - 1.0
                ),
            }
            for label, cand in cand_sets.items():
                emb = (
                    emb_init.unsqueeze(0).unsqueeze(0)
                    .expand(1, n_cand, *emb_init.shape).contiguous()
                )
                out = jepa.rollout({"emb": emb}, cand.unsqueeze(0), history_size=H)
                z = out["predicted_emb"][0, :, -1].reshape(n_cand, -1).float()
                spread = float((z - z.mean(0, keepdim=True)).norm(dim=1).mean())
                fwd = float((z - z_init).norm(dim=1).mean())
                acc[f"{label}_spread"].append(spread)
                acc[f"{label}_fwd"].append(fwd)
                acc[f"{label}_ratio"].append(spread / (fwd + 1e-9))
                if label == "local":
                    costs = (z - z_goal).pow(2).mean(dim=1)
                    acc["cost_cv"].append(float(costs.std() / (costs.mean() + 1e-9)))

    summary: dict[str, Any] = {
        k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()
    }
    summary["n_episodes"] = len(acc["traj_std"])
    return summary


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
    ap.add_argument(
        "--mode",
        choices=["rollout", "action-sens"],
        default="rollout",
        help="rollout = Q1a rollout-accuracy-vs-persistence; action-sens = "
        "latent action-sensitivity diagnostic (do different action sequences "
        "change the predicted terminal latent?).",
    )
    ap.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Rollout horizon for --mode action-sens (default: cfg max_horizon).",
    )
    ap.add_argument(
        "--n-cand",
        type=int,
        default=64,
        help="Candidate action sequences per state for --mode action-sens.",
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

    out_dir = (
        Path(args.out_dir).expanduser()
        if args.out_dir is not None
        else Path(cfg.get("out_dir", "/tmp/vitruvian/rollout_eval")).expanduser()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "action-sens":
        horizon = (
            args.horizon if args.horizon is not None
            else int(cfg.get("max_horizon", 16))
        )
        res = action_sensitivity(
            jepa, dataset.patches, dataset.ep_offset, dataset.ep_len,
            dataset.action, episodes,
            history_size=history_size, horizon=horizon,
            device=device, n_cand=args.n_cand, seed=int(cfg.get("seed", 0)),
        )
        print(
            f"[action-sens] held-out episodes: {res['n_episodes']}  "
            f"(horizon={horizon}, n_cand={args.n_cand})"
        )
        print(
            f"  local  : spread={res['local_spread']:.4f}  "
            f"fwd={res['local_fwd']:.4f}  spread/fwd={res['local_ratio']:.4f}"
        )
        print(
            f"  diverse: spread={res['diverse_spread']:.4f}  "
            f"fwd={res['diverse_fwd']:.4f}  spread/fwd={res['diverse_ratio']:.4f}"
        )
        print(
            f"  trajectory latent std={res['traj_std']:.4f} | "
            f"MPPI cost CV (local)={res['cost_cv']:.4f}"
        )
        verdict = (
            "LOW action-sensitivity — steering-limited (motivates LDAD)"
            if res["diverse_ratio"] < 0.15
            else "action-sensitive"
        )
        print(f"[action-sens] verdict: {verdict}")
        (out_dir / "action_sensitivity.json").write_text(json.dumps(res, indent=2))
        print(f"[action-sens] wrote {out_dir / 'action_sensitivity.json'}")
        return

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
