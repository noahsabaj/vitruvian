# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""MPPIPlanner action/history alignment + history-size edge cases.

These guard the plan-time action convention: the history-action block
carries ``history_size - 1`` actions so the first planned action ``U[0]``
occupies the current (last context) frame's slot, matching the
``(s_t, a_t)`` convention the predictor was trained on.
"""

from __future__ import annotations

import torch

from vitruvian.models import PlannerBackbone, build_jepa
from vitruvian.planning import MPPIPlanner, MSECost


def _v5_jepa():
    cfg = {
        "backbone": {"name": "dinov3-patch", "kwargs": {}},
        "predictor": {
            "name": "patch-ar",
            "kwargs": {
                "num_frames": 3, "depth": 1, "heads": 2, "mlp_dim": 64,
                "input_dim": 32, "hidden_dim": 32, "output_dim": 32,
                "dim_head": 16, "dropout": 0.0, "adaln_rank": 16,
            },
        },
        "action_encoder": {
            "kwargs": {"action_dim": 29, "frameskip": 1, "smoothed_dim": 10,
                        "emb_dim": 32}
        },
        "patch_projector": {"kwargs": {"in_dim": 768, "out_dim": 32}},
    }
    return build_jepa(cfg)


def _plan_with_hs(hs: int) -> torch.Tensor:
    jepa = _v5_jepa()
    pb = PlannerBackbone(jepa)
    planner = MPPIPlanner(
        jepa=jepa, backbone=pb, subgoal_emb=torch.randn(49, 32),
        cost_fn=MSECost(), action_dim=29, horizon=5, num_samples=4,
        noise_sigma=0.02, iterations=1, history_size=hs, device="cpu",
    )
    return planner.plan(
        pixel_history=None,
        action_history=torch.zeros(2, 29),
        encoded_history=torch.randn(hs, 49, 32),
    )


def test_plan_hs3(registry_with_fakes) -> None:
    U = _plan_with_hs(3)
    assert U.shape == (5, 29)
    assert torch.isfinite(U).all()


def test_plan_hs1_no_history_actions(registry_with_fakes) -> None:
    # history_size == 1 → zero history actions; U[0] is the only context
    # action. Guards the ``n_hist_act == 0`` branch (and the ``ah[-0:]``
    # foot-gun that a naive ``ah[-(HS-1):]`` slice would hit).
    U = _plan_with_hs(1)
    assert U.shape == (5, 29)
    assert torch.isfinite(U).all()
