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


def test_prediction_loss_target_is_1_step_tf() -> None:
    """Semantic guard: the TF target must be ``emb[:, 1:ctx_len+1]``,
    not ``emb[:, n_preds:n_preds+ctx_len]`` (the inherited-LeWM offset
    that M4.9.1 fixed).

    Setup: synthetic embedding where ``emb[t] = alpha * t * ones(D)``.
    An identity predictor ``predict(ctx, _) = ctx`` produces the
    context itself as its "prediction." Under correct 1-step TF
    semantics, the target at position t is ``emb[t+1] = alpha*(t+1)``,
    so the residual is exactly ``-alpha`` per dim, and the MSE is
    exactly ``alpha^2``. Under the inherited offset the target would
    be ``emb[t + n_preds]``, giving MSE ``(alpha * n_preds)^2`` — way
    off. The assertion below catches either regression.
    """
    alpha = 0.7
    B, T, D = 2, 9, 4  # T = ctx_len + num_preds = 3 + 6
    ctx_len, num_preds = 3, 6

    # emb[b, t, d] = alpha * t (same for every batch + every dim).
    t_grid = torch.arange(T, dtype=torch.float32).view(1, T, 1)
    emb = alpha * t_grid.expand(B, T, D).clone()

    class IdentityModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proprio_encoder = None
            self.patch_projector = None

            class _ActE(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.l = nn.Linear(1, D)

                def forward(self, x):
                    return self.l(x.float().mean(dim=-1, keepdim=True))

            self.action_encoder = _ActE()

        def predict(self, emb_in, act_in):
            # Identity predictor — matches the synthetic assumption above.
            return emb_in

    batch = {
        "emb": emb,
        "proprio": torch.zeros(B, T, 0),  # no proprio
        "action": torch.zeros(B, T, 1),
    }
    info = prediction_loss(
        IdentityModel(), batch,
        history_size=ctx_len, num_preds=num_preds,
        rollout_weight=0.0, std_weight=0.0,
    )
    # Under correct 1-step TF: pred_loss = alpha^2.
    # Under inherited (n_preds-offset) broken path: pred_loss = (alpha * n_preds)^2.
    assert torch.allclose(
        info["pred_loss"],
        torch.tensor(alpha**2),
        atol=1e-5,
    ), (
        f"pred_loss is {info['pred_loss'].item():.4f}; expected {alpha**2:.4f} "
        f"(alpha^2). If you got ~{(alpha * num_preds) ** 2:.4f}, someone "
        f"re-introduced the inherited n_preds target offset."
    )


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
