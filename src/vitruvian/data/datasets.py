# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Datasets over the G1 expert HDF5.

Three dataset classes, all keyed off the same precomputed embedding
cache (see :class:`vitruvian.data.EmbeddingCache`):

* :class:`G1EmbSeqDataset` — CLS-shaped fixed-length sequences for v4
  JEPA training. Yields ``(emb, proprio, action)`` windows.
* :class:`G1PatchSeqDataset` — patch-shaped windows for v5 JEPA.
  Yields ``(patches, proprio, action)``.
* :class:`G1HERTransitionDataset` — HER-relabeled (``s, s', g``)
  transitions for value-head training. Goal is sampled uniformly from
  future frames within the same episode.

Pixel data is never loaded — all modules consume precomputed
embeddings. Proprio + action arrays are materialized in host RAM at
``__init__`` (~100 MB for 270k × (103+29) float32).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _read_meta(h5_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with h5py.File(h5_path, "r") as f:
        ep_offset = f["ep_offset"][:].astype(np.int64)
        ep_len = f["ep_len"][:].astype(np.int64)
        n_total = int(f["pixels"].shape[0])
    return ep_offset, ep_len, n_total


def _read_proprio_action(h5_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(h5_path, "r") as f:
        proprio = torch.from_numpy(f["proprio"][:]).float()
        action = torch.from_numpy(f["action"][:]).float()
    return proprio, action


def _valid_seq_starts(
    ep_offset: np.ndarray, ep_len: np.ndarray, seq_len: int
) -> np.ndarray:
    valid: list[int] = []
    for o, L in zip(ep_offset, ep_len):
        for t in range(int(o), int(o + L - seq_len + 1)):
            valid.append(t)
    return np.asarray(valid, dtype=np.int64)


class G1EmbSeqDataset(Dataset[dict[str, Any]]):
    """CLS-shaped fixed-length windows for v4 training.

    Yields per-sample dicts::

        {
          "emb":     (seq_len, D_emb)      float32,
          "proprio": (seq_len, D_prop)     float32,
          "action":  (seq_len, action_dim) float32,
        }

    where ``seq_len = history_size + num_preds``.
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

        ep_offset, ep_len, n_total = _read_meta(self.h5_path)
        self.proprio, self.action = _read_proprio_action(self.h5_path)

        assert emb_cache.shape[0] == n_total, (
            f"emb cache rows {emb_cache.shape[0]} != H5 rows {n_total}"
        )
        self.emb = emb_cache
        self.ep_offset = ep_offset
        self.ep_len = ep_len
        self.valid_idx = _valid_seq_starts(ep_offset, ep_len, self.seq_len)

    def __len__(self) -> int:
        return len(self.valid_idx)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        t0 = int(self.valid_idx[idx])
        s = slice(t0, t0 + self.seq_len)
        return {
            "emb": self.emb[s],
            "proprio": self.proprio[s],
            "action": self.action[s],
        }


class G1PatchSeqDataset(Dataset[dict[str, Any]]):
    """Patch-shaped fixed-length windows for v5 training.

    Yields per-sample dicts::

        {
          "patches": (seq_len, N_patches, patch_dim)  fp16
          "proprio": (seq_len, D_prop)                fp32
          "action":  (seq_len, action_dim)            fp32
        }

    Patches stay fp16 in RAM to halve the footprint; cast to fp32 on
    the GPU per-batch in the train loop.
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

        ep_offset, ep_len, n_total = _read_meta(self.h5_path)
        self.proprio, self.action = _read_proprio_action(self.h5_path)

        assert patch_cache.shape[0] == n_total, (
            f"patch cache rows {patch_cache.shape[0]} != H5 rows {n_total}"
        )
        self.patches = patch_cache
        self.ep_offset = ep_offset
        self.ep_len = ep_len
        self.valid_idx = _valid_seq_starts(ep_offset, ep_len, self.seq_len)

    def __len__(self) -> int:
        return len(self.valid_idx)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        t0 = int(self.valid_idx[idx])
        s = slice(t0, t0 + self.seq_len)
        return {
            "patches": self.patches[s],
            "proprio": self.proprio[s],
            "action": self.action[s],
        }


class G1HERTransitionDataset(Dataset[dict[str, Any]]):
    """HER-relabeled ``(s, s', g)`` transitions for VF training.

    For each ``(t, t+1)`` pair within an episode, samples a goal index
    ``g ∈ [t+1, ep_end)`` uniformly at random. This gives the IQL loss
    genuine goal-reaching structure (cross-trajectory goals don't).

    ``is_goal_reached`` flags the transitions whose *next* state is the
    sampled goal (``t + 1 == g``). These are the terminal transitions of
    the relabeled goal-reaching MDP: the consumer (:class:`~vitruvian.
    training.iql.VFHERTrainer`) gives them the ``reward_self_loop``
    reward and drops the bootstrap, which is what grounds the value
    head's ``V(g, g) ≈ 0`` anchor. Sampling from ``[t+1, ep_end)``
    guarantees a non-empty set of such transitions (the last in-episode
    pair always has ``g == t + 1``). The earlier ``is_goal_self =
    (t == g)`` flag was dead — ``g`` is never ``t`` — so no transition
    was ever terminal and the value target degenerated to a constant.

    Yields per-sample dicts::

        {
          "emb_t":          (D_emb,) float32,
          "prop_t":         (D_prop,) float32,
          "emb_tp1":        (D_emb,) float32,
          "prop_tp1":       (D_prop,) float32,
          "emb_g":          (D_emb,) float32,
          "prop_g":         (D_prop,) float32,
          "is_goal_reached": bool tensor (t + 1 == g),
        }
    """

    def __init__(
        self,
        h5_path: Path,
        emb_cache: torch.Tensor,
        *,
        seed: int = 0,
    ) -> None:
        self.h5_path = Path(h5_path)
        ep_offset, ep_len, n_total = _read_meta(self.h5_path)
        self.proprio, _ = _read_proprio_action(self.h5_path)

        assert emb_cache.shape[0] == n_total, (
            f"emb cache rows {emb_cache.shape[0]} != H5 rows {n_total}"
        )
        self.emb = emb_cache

        rng = np.random.default_rng(seed)
        valid_t: list[int] = []
        goal_g: list[int] = []
        for o, L in zip(ep_offset, ep_len):
            o_i, L_i = int(o), int(L)
            ep_end_exclusive = o_i + L_i
            for t in range(o_i, o_i + L_i - 1):
                g = int(rng.integers(t + 1, ep_end_exclusive))
                valid_t.append(t)
                goal_g.append(g)
        self.valid_idx = np.asarray(valid_t, dtype=np.int64)
        self.goal_idx = np.asarray(goal_g, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.valid_idx)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        t = int(self.valid_idx[idx])
        g = int(self.goal_idx[idx])
        return {
            "emb_t": self.emb[t],
            "prop_t": self.proprio[t],
            "emb_tp1": self.emb[t + 1],
            "prop_tp1": self.proprio[t + 1],
            "emb_g": self.emb[g],
            "prop_g": self.proprio[g],
            # Terminal transition of the relabeled MDP: the next state is
            # the goal. ``g`` is sampled from ``[t+1, ep_end)``, so this
            # is the achievable "goal reached" event (``t == g`` never is).
            "is_goal_reached": torch.tensor(t + 1 == g, dtype=torch.bool),
        }


def episode_aware_split(
    ep_offset: np.ndarray,
    valid_idx: np.ndarray,
    *,
    val_frac: float,
    seed: int,
    seq_len: int,
) -> tuple[list[int], list[int]]:
    """Split window indices into ``(train, val)`` WITHOUT frame leakage.

    Sliding windows from the same episode overlap by ``seq_len - 1``
    frames, so a per-window random split (``random_split``) scatters
    near-duplicate windows across train and val and inflates the
    validation metric. This splits by *whole episode* instead — every
    window of an episode goes to the same side — so no frame is shared.

    Episodes are shuffled with ``seed`` and assigned to val until about
    ``val_frac`` of the windows are covered, always leaving at least one
    episode for train (and one window for val). With a single episode it
    falls back to a contiguous split with a ``seq_len`` gap, so the two
    halves still share no frame.

    Args:
        ep_offset: ``(n_episodes,)`` per-episode start row (ascending).
        valid_idx: ``(n_windows,)`` window-start rows, as built by the
            dataset (sorted; each window lies within one episode).
        val_frac: target fraction of windows for validation.
        seed: RNG seed for episode shuffling.
        seq_len: window length (used for the single-episode gap).

    Returns:
        ``(train_idx, val_idx)`` — lists of indices into ``valid_idx``.
    """
    n_win = len(valid_idx)
    if n_win == 0:
        return [], []

    ep_id = np.searchsorted(ep_offset, valid_idx, side="right") - 1
    uniq = np.unique(ep_id)

    if len(uniq) < 2:
        # Single episode: contiguous split with a seq_len gap so train and
        # val windows share no frame.
        n_val = max(1, int(round(val_frac * n_win)))
        val_start = n_win - n_val
        train_end = max(0, val_start - (seq_len - 1))
        return list(range(train_end)), list(range(val_start, n_win))

    rng = np.random.default_rng(seed)
    order = uniq[rng.permutation(len(uniq))]
    counts = {int(e): int((ep_id == e).sum()) for e in uniq}
    target_val = max(1, int(round(val_frac * n_win)))

    val_eps: set[int] = set()
    cum = 0
    for e in order:
        if cum >= target_val or len(val_eps) >= len(uniq) - 1:
            break
        val_eps.add(int(e))
        cum += counts[int(e)]

    train_idx = [i for i in range(n_win) if int(ep_id[i]) not in val_eps]
    val_idx = [i for i in range(n_win) if int(ep_id[i]) in val_eps]
    return train_idx, val_idx


__all__ = [
    "G1EmbSeqDataset",
    "G1HERTransitionDataset",
    "G1PatchSeqDataset",
    "episode_aware_split",
]
