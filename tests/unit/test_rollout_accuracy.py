# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Q1a open-loop rollout-accuracy metric — CPU wiring + alignment tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vitruvian.cli.rollout import action_sensitivity, rollout_accuracy
from vitruvian.data import G1PatchSeqDataset
from vitruvian.models import build_jepa


def _v5_cfg() -> dict:
    # Mirror tests/unit/test_jepa.py::_v5_cfg — small patch JEPA (hidden 128,
    # 49 patches) built on the fake patch backbone (raw dim 768).
    return {
        "backbone": {"name": "dinov3-patch", "kwargs": {}},
        "predictor": {
            "name": "patch-ar",
            "kwargs": {
                "num_frames": 3, "depth": 2, "heads": 4,
                "mlp_dim": 256, "input_dim": 128, "hidden_dim": 128,
                "output_dim": 128, "dim_head": 32, "dropout": 0.0,
                "adaln_rank": 64,
            },
        },
        "action_encoder": {
            "kwargs": {"action_dim": 29, "frameskip": 1, "smoothed_dim": 10}
        },
        "proprio_encoder": {
            "kwargs": {
                "in_dim": 103, "hidden": 64, "out_dim": 128, "norm_fn": None
            }
        },
        "patch_projector": {"kwargs": {"in_dim": 768, "out_dim": 128}},
    }


def _dataset(synthetic_h5) -> G1PatchSeqDataset:
    # Fake raw patch cache matching the fake backbone's 768-d, 49-patch output.
    total = 50  # 2 episodes x 25 steps (see conftest.synthetic_h5)
    cache = torch.randn(total, 49, 768)
    return G1PatchSeqDataset(synthetic_h5, cache, seq_len=9)


def test_rollout_accuracy_structural(registry_with_fakes, synthetic_h5) -> None:
    m = build_jepa(_v5_cfg())
    ds = _dataset(synthetic_h5)
    res = rollout_accuracy(
        m, ds.patches, ds.ep_offset, ds.ep_len, ds.action, [0, 1],
        history_size=3, max_horizon=10, device="cpu",
    )
    assert res["n_episodes"] == 2
    # ep_len 25, H=3 -> K = min(10, 22) = 10 horizons, each with both episodes.
    assert len(res["per_horizon"]) == 10
    for row in res["per_horizon"]:
        assert row["n"] == 2
        assert -1.0 <= row["model_cos"] <= 1.0
        assert -1.0 <= row["persist_cos"] <= 1.0
        assert row["model_mse"] >= 0.0 and row["persist_mse"] >= 0.0
        assert np.isfinite(row["model_cos"]) and np.isfinite(row["model_mse"])
    assert isinstance(res["summary"]["beats_persistence"], bool)


def test_rollout_accuracy_persistence_predictor(
    registry_with_fakes, synthetic_h5
) -> None:
    """With an identity predictor the rollout copies the last seed frame at
    every step, so its predictions are *exactly* the persistence baseline.
    This pins the metric's horizon indexing + projection alignment: if the
    model's scores did not equal persistence, the future frames would be
    mis-indexed (e.g. reading into the given history)."""
    m = build_jepa(_v5_cfg())
    m.predict = lambda emb, cond: emb  # identity -> rollout appends last frame
    ds = _dataset(synthetic_h5)
    res = rollout_accuracy(
        m, ds.patches, ds.ep_offset, ds.ep_len, ds.action, [0, 1],
        history_size=3, max_horizon=8, device="cpu",
    )
    assert res["per_horizon"]
    for row in res["per_horizon"]:
        assert row["model_cos"] == pytest.approx(row["persist_cos"], abs=1e-5)
        assert row["model_mse"] == pytest.approx(row["persist_mse"], abs=1e-5)
    assert not res["summary"]["beats_persistence"]  # identical -> not strictly >


def test_action_sensitivity_structural(registry_with_fakes, synthetic_h5) -> None:
    m = build_jepa(_v5_cfg())
    ds = _dataset(synthetic_h5)
    res = action_sensitivity(
        m, ds.patches, ds.ep_offset, ds.ep_len, ds.action, [0, 1],
        history_size=3, horizon=8, device="cpu", n_cand=16,
    )
    assert res["n_episodes"] == 2
    for k in (
        "local_spread", "local_fwd", "local_ratio", "diverse_spread",
        "diverse_fwd", "diverse_ratio", "cost_cv", "traj_std",
    ):
        assert np.isfinite(res[k]), f"{k} not finite"
    assert res["local_spread"] >= 0.0 and res["diverse_spread"] >= 0.0
    assert res["local_ratio"] >= 0.0 and res["diverse_ratio"] >= 0.0
