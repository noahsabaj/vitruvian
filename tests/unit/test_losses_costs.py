# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Training losses + planning costs."""

from __future__ import annotations

import torch
import torch.nn as nn

from vitruvian.planning import MSECost, PatchMSECost, ValueHeadCost
from vitruvian.training import prediction_loss, vicreg_std_loss
from vitruvian.training.iql import ValueHead


def test_vicreg_std_loss_identifies_collapse() -> None:
    collapsed = torch.randn(4, 5, 32) * 1e-4
    spread = torch.randn(4, 5, 32) * 2.0
    assert vicreg_std_loss(collapsed) > vicreg_std_loss(spread)


def test_mse_cost_flat() -> None:
    pred = torch.randn(16, 64)
    goal = torch.zeros(64)
    c = MSECost()(pred, goal)
    assert c.shape == (16,)
    assert torch.allclose(c, pred.pow(2).sum(-1))


def test_mse_cost_patches() -> None:
    pred = torch.randn(16, 49, 64)
    goal = torch.zeros(49, 64)
    c = MSECost()(pred, goal)
    assert c.shape == (16,)
    assert torch.allclose(c, pred.pow(2).flatten(1).sum(-1))


def test_patch_mse_cost_normalization() -> None:
    pred = torch.randn(8, 49, 64)
    goal = torch.zeros(49, 64)
    std = torch.ones(49, 64) * 2.0
    c1 = PatchMSECost()(pred, goal)
    c2 = PatchMSECost(patch_std=std)(pred, goal)
    # normalization ≈ division by std+eps ≈ 2 → cost scales by 1/4.
    assert torch.all(c2 < c1)


def test_value_head_cost_rejects_patches() -> None:
    vh = ValueHead(emb_dim=64, hidden=32, out_dim=16)
    patched = torch.randn(4, 49, 64)
    goal = torch.zeros(49, 64)
    import pytest

    with pytest.raises(RuntimeError):
        ValueHeadCost(vh)(patched, goal)


def test_prediction_loss_patches_runs() -> None:
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_projector = nn.Linear(768, 64)
            self.proprio_encoder = nn.Linear(103, 64)

            class _ActE(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.l = nn.Linear(29, 64)

                def forward(self, x):
                    return self.l(x.float())

            self.action_encoder = _ActE()

        def predict(self, emb, act):
            return emb

    model = FakeModel()
    B, T, N = 2, 6, 49
    batch = {
        "patches": torch.randn(B, T, N, 768),
        "proprio": torch.randn(B, T, 103),
        "action": torch.randn(B, T, 29),
    }
    info = prediction_loss(
        model, batch, history_size=3, num_preds=3,
        rollout_weight=1.0, std_weight=0.5,
    )
    assert torch.isfinite(info["loss"])
    assert torch.isfinite(info["pred_loss"])
    assert torch.isfinite(info["std_loss"])
