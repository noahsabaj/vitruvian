# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Test-time adapter (AdaJEPA) — CPU tests for the thesis-safety boundary
(only the predictor moves) and that adaptation actually learns."""

from __future__ import annotations

import torch

from vitruvian.models import build_jepa
from vitruvian.planning.adapt import TestTimeAdapter


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


def _fill(adapter: TestTimeAdapter, n: int = 6) -> None:
    torch.manual_seed(0)
    for _ in range(n):
        adapter.push(torch.randn(49, 128), torch.randn(29))


def test_adapter_updates_predictor_only(registry_with_fakes) -> None:
    m = build_jepa(_v5_cfg())
    adapter = TestTimeAdapter(m, history_size=3, num_preds=1, lr=1e-3)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}

    _fill(adapter)
    l1 = adapter.step(n_steps=1)
    l_more = adapter.step(n_steps=25)
    after = dict(m.named_parameters())

    # The predictor moved.
    assert any(
        not torch.allclose(before[n], after[n])
        for n in before
        if n.startswith("predictor.")
    )
    # The frozen prior, the projector, and the (unoptimized) encoders did not
    # — this is the thesis-safety boundary.
    for n in before:
        if n.split(".", 1)[0] in {
            "backbone", "patch_projector", "action_encoder", "proprio_encoder"
        }:
            assert torch.allclose(before[n], after[n]), f"{n} must stay frozen"

    # Adaptation actually reduces prediction error on the buffered window.
    assert l1 is not None and l_more is not None
    assert l_more < l1


def test_adapter_reset_restores(registry_with_fakes) -> None:
    m = build_jepa(_v5_cfg())
    adapter = TestTimeAdapter(m, history_size=3, num_preds=1, lr=1e-3)
    before = {
        n: p.detach().clone()
        for n, p in m.named_parameters()
        if n.startswith("predictor.")
    }
    _fill(adapter)
    adapter.step(n_steps=10)
    adapter.reset()
    after = dict(m.named_parameters())
    for n, saved in before.items():
        assert torch.allclose(saved, after[n], atol=1e-6), f"{n} not restored"


def test_adapter_noop_until_full_window(registry_with_fakes) -> None:
    m = build_jepa(_v5_cfg())
    adapter = TestTimeAdapter(m, history_size=3, num_preds=1)  # seq_len 4
    adapter.push(torch.randn(49, 128), torch.randn(29))
    assert adapter.step(n_steps=1) is None  # only 1 < 4 transitions buffered


def _v6_cfg() -> dict:
    cfg = _v5_cfg()
    cfg["predictor"] = {
        "name": "prefix-patch",
        "kwargs": {
            "depth": 2, "heads": 4, "mlp_dim": 256, "input_dim": 128,
            "hidden_dim": 128, "output_dim": 128, "dim_head": 32,
            "prefix_depth": 2, "dropout": 0.0, "adaln_rank": 64,
            "max_horizon": 8,
        },
    }
    return cfg


def test_adapter_prefix_predictor_runs(registry_with_fakes) -> None:
    """A Fast-LeWM (prefix) checkpoint must adapt via the dense prefix loss and
    with history_size forced to 1 — not crash on the AR-only prediction_loss /
    3-D anchor assertion (the M6 adapter bug)."""
    m = build_jepa(_v6_cfg())
    adapter = TestTimeAdapter(m, history_size=3, num_preds=1, lr=1e-3)
    assert adapter.is_prefix and adapter.history_size == 1  # forced for prefix

    before = {
        n: p.detach().clone()
        for n, p in m.named_parameters()
        if n.startswith("predictor.")
    }
    _fill(adapter, n=4)
    loss = adapter.step(n_steps=3)
    assert loss is not None
    after = dict(m.named_parameters())
    assert any(
        not torch.allclose(before[n], after[n]) for n in before
    ), "prefix adapter must update the predictor"
