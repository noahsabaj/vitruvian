# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Dataset classes over the synthetic HDF5."""

from __future__ import annotations

from pathlib import Path

import torch

import numpy as np

from vitruvian.data import (
    G1EmbSeqDataset,
    G1HERTransitionDataset,
    G1PatchSeqDataset,
    episode_aware_split,
)


def _frames(valid_idx, idxs, seq_len: int) -> set[int]:
    """Set of cache rows covered by the given windows."""
    s: set[int] = set()
    for j in idxs:
        start = int(valid_idx[j])
        s.update(range(start, start + seq_len))
    return s


def test_emb_seq_dataset(synthetic_h5: Path) -> None:
    # 2 episodes × 25 steps → 50 total; seq_len 9 → 17 valid starts per ep.
    emb = torch.zeros(50, 768)
    ds = G1EmbSeqDataset(synthetic_h5, emb, seq_len=9)
    assert len(ds) == 2 * (25 - 9 + 1)
    sample = ds[0]
    assert sample["emb"].shape == (9, 768)
    assert sample["proprio"].shape == (9, 103)
    assert sample["action"].shape == (9, 29)


def test_patch_seq_dataset(synthetic_h5: Path) -> None:
    patches = torch.zeros(50, 49, 768, dtype=torch.float16)
    ds = G1PatchSeqDataset(synthetic_h5, patches, seq_len=6)
    assert len(ds) == 2 * (25 - 6 + 1)
    sample = ds[0]
    assert sample["patches"].shape == (6, 49, 768)
    assert sample["patches"].dtype == torch.float16


def test_her_transition_dataset(synthetic_h5: Path) -> None:
    emb = torch.randn(50, 768)
    ds = G1HERTransitionDataset(synthetic_h5, emb, seed=0)
    # Each (t, t+1) pair — 24 per 25-step episode.
    assert len(ds) == 2 * 24
    s = ds[0]
    assert s["emb_t"].shape == (768,)
    assert s["prop_t"].shape == (103,)
    assert s["emb_g"].shape == (768,)
    # HER: goal index g >= t+1 in same episode.
    assert ds.goal_idx[0] > ds.valid_idx[0]
    # The terminal flag fires exactly when the next state is the goal
    # (t + 1 == g), and at least the last in-episode pair must qualify.
    reached = torch.stack([ds[i]["is_goal_reached"] for i in range(len(ds))])
    expected = torch.from_numpy(ds.goal_idx == ds.valid_idx + 1)
    assert torch.equal(reached, expected)
    assert reached.any()


def test_episode_aware_split_no_frame_leakage(synthetic_h5: Path) -> None:
    seq_len = 9
    ds = G1EmbSeqDataset(synthetic_h5, torch.zeros(50, 768), seq_len=seq_len)
    train_idx, val_idx = episode_aware_split(
        ds.ep_offset, ds.valid_idx, val_frac=0.5, seed=0, seq_len=seq_len
    )
    assert train_idx and val_idx
    # Every window assigned exactly once.
    assert sorted(train_idx + val_idx) == list(range(len(ds)))
    # No cache row appears in both splits (this is the leakage the old
    # random_split allowed).
    tr = _frames(ds.valid_idx, train_idx, seq_len)
    va = _frames(ds.valid_idx, val_idx, seq_len)
    assert tr.isdisjoint(va)


def test_episode_aware_split_single_episode_has_gap() -> None:
    # One episode, 10 frames → 7 windows of length 4. The fallback must
    # still produce frame-disjoint train/val via a seq_len gap.
    seq_len = 4
    ep_offset = np.array([0], dtype=np.int64)
    valid_idx = np.arange(0, 7, dtype=np.int64)
    train_idx, val_idx = episode_aware_split(
        ep_offset, valid_idx, val_frac=0.3, seed=0, seq_len=seq_len
    )
    assert train_idx and val_idx
    tr = _frames(valid_idx, train_idx, seq_len)
    va = _frames(valid_idx, val_idx, seq_len)
    assert tr.isdisjoint(va)
