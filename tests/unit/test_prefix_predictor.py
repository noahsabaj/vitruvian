# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Fast-LeWM PrefixPatchPredictor (M6) — shape + registry-build tests."""

from __future__ import annotations

import pytest
import torch

from vitruvian.models import build_jepa
from vitruvian.models.predictors import PrefixPatchPredictor


def _pred(**over) -> PrefixPatchPredictor:
    kw = dict(
        num_patches=49, depth=2, heads=4, mlp_dim=256, input_dim=128,
        hidden_dim=128, output_dim=128, dim_head=32, prefix_depth=2,
        adaln_rank=64, max_horizon=8,
    )
    kw.update(over)
    return PrefixPatchPredictor(**kw)


def test_prefix_predictor_shapes() -> None:
    m = _pred()
    B, N, H = 2, 49, 5
    out = m(torch.randn(B, N, 128), torch.randn(B, H, 128))
    assert out.shape == (B, H, N, 128)  # one predicted latent per horizon


def test_prefix_predictor_variable_horizon() -> None:
    # Same module evaluates different horizons (up to max_horizon) — the whole
    # point of prefix prediction.
    m = _pred(max_horizon=8)
    for h in (1, 3, 8):
        out = m(torch.randn(1, 49, 128), torch.randn(1, h, 128))
        assert out.shape == (1, h, 49, 128)
    with pytest.raises(ValueError):
        m(torch.randn(1, 49, 128), torch.randn(1, 9, 128))  # > max_horizon


def test_prefix_encoder_is_causal() -> None:
    """The action-prefix encoder must be causal: prefix token k summarizes only
    a_0..a_{k-1}, so perturbing a later action changes only that horizon's prefix
    (tested at the encoder, not the full predictor — the predictor's AdaLN head
    is zero-init, so an untrained predictor is action-independent by design)."""
    m = _pred().eval()
    state = torch.randn(1, 128)
    act = torch.randn(1, 5, 128)
    with torch.no_grad():
        base = m._encode_prefixes(state, act)  # (1, 5, hidden)
        act2 = act.clone()
        act2[:, 4] += 5.0  # perturb only the LAST action (a_4)
        pert = m._encode_prefixes(state, act2)
    # Prefixes 1..4 see only a_0..a_3 -> unchanged; prefix 5 sees a_4 -> changes.
    assert torch.allclose(base[:, :4], pert[:, :4], atol=1e-5)
    assert not torch.allclose(base[:, 4], pert[:, 4])


def test_build_prefix_patch_via_registry(registry_with_fakes) -> None:
    cfg = {
        "backbone": {"name": "dinov3-patch", "kwargs": {}},
        "predictor": {
            "name": "prefix-patch",
            "kwargs": {
                "depth": 2, "heads": 4, "mlp_dim": 256, "input_dim": 128,
                "hidden_dim": 128, "output_dim": 128, "dim_head": 32,
                "prefix_depth": 2, "adaln_rank": 64, "max_horizon": 8,
            },
        },
        "action_encoder": {
            "kwargs": {"action_dim": 29, "frameskip": 1, "smoothed_dim": 10}
        },
        "proprio_encoder": {
            "kwargs": {"in_dim": 103, "hidden": 64, "out_dim": 128, "norm_fn": None}
        },
        "patch_projector": {"kwargs": {"in_dim": 768, "out_dim": 128}},
    }
    m = build_jepa(cfg)
    assert m.predictor.num_patches == 49  # injected from the patch backbone
    assert m.emb_dim == 128
