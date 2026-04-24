# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Unified JEPA composer — build, encode, rollout shapes."""

from __future__ import annotations

import torch

from vitruvian.models import JEPA, PlannerBackbone, build_jepa


def _v5_cfg() -> dict:
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


def _v4_cfg() -> dict:
    return {
        "backbone": {"name": "dinov3-cls", "kwargs": {}},
        "predictor": {
            "name": "ar",
            "kwargs": {
                "num_frames": 3, "depth": 2, "heads": 4,
                "mlp_dim": 256, "input_dim": 768, "hidden_dim": 768,
                "output_dim": 768, "dim_head": 64, "dropout": 0.0,
            },
        },
        "action_encoder": {
            "kwargs": {"action_dim": 29, "frameskip": 1, "smoothed_dim": 10}
        },
        "proprio_encoder": {
            "kwargs": {
                "in_dim": 103, "hidden": 128, "out_dim": 768, "norm_fn": None
            }
        },
    }


def test_build_v5_encode_shape(registry_with_fakes) -> None:
    m = build_jepa(_v5_cfg())
    assert isinstance(m, JEPA)
    assert m.n_patches == 49
    assert m.emb_dim == 128
    B, T = 2, 3
    info = {
        "pixels": torch.zeros(B, T, 3, 224, 224),
        "proprio": torch.randn(B, T, 103),
        "action": torch.randn(B, T, 29),
    }
    out = m.encode(info)
    assert out["emb"].shape == (B, T, 49, 128)
    assert out["act_emb"].shape == (B, T, 128)


def test_build_v4_encode_shape(registry_with_fakes) -> None:
    m = build_jepa(_v4_cfg())
    assert m.patch_projector is None
    B, T = 2, 3
    out = m.encode(
        {
            "pixels": torch.zeros(B, T, 3, 224, 224),
            "proprio": torch.randn(B, T, 103),
            "action": torch.randn(B, T, 29),
        }
    )
    assert out["emb"].shape == (B, T, 768)


def test_rollout_accepts_pre_encoded(registry_with_fakes) -> None:
    m = build_jepa(_v5_cfg())
    B, S, T_hist, T_future = 1, 2, 3, 2
    emb_pre = torch.randn(B, S, T_hist, 49, 128)
    acts = torch.randn(B, S, T_hist + T_future, 29)
    out = m.rollout({"emb": emb_pre}, acts, history_size=3)
    assert out["predicted_emb"].shape == (
        B, S, T_hist + T_future + 1, 49, 128
    )


def test_planner_backbone_encode(registry_with_fakes) -> None:
    m = build_jepa(_v5_cfg())
    pb = PlannerBackbone(m)
    assert pb.output_dim == 128
    assert pb.n_patches == 49
    out = pb.encode(torch.zeros(1, 1, 3, 224, 224))
    assert out.shape == (1, 1, 49, 128)
