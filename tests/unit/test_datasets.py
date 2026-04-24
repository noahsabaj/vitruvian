# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Dataset classes over the synthetic HDF5."""

from __future__ import annotations

from pathlib import Path

import torch

from vitruvian.data import (
    G1EmbSeqDataset,
    G1HERTransitionDataset,
    G1PatchSeqDataset,
)


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
