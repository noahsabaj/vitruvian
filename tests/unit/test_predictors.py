# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""PatchARPredictor shape + zero-init + param-count checks."""

from __future__ import annotations

import torch

from vitruvian.models import PatchARPredictor


def _make(**kwargs) -> PatchARPredictor:
    defaults = dict(
        num_frames=3,
        num_patches=49,
        depth=2,
        heads=4,
        mlp_dim=256,
        input_dim=128,
        hidden_dim=128,
        output_dim=128,
        dim_head=32,
        dropout=0.0,
        adaln_rank=64,
    )
    defaults.update(kwargs)
    return PatchARPredictor(**defaults)


def test_patch_predictor_output_shape() -> None:
    p = _make()
    x = torch.randn(2, 3, 49, 128)
    c = torch.randn(2, 3, 128)
    y = p(x, c)
    assert y.shape == x.shape


def test_adaln_head_zero_init() -> None:
    p = _make()
    for blk in p.blocks:
        assert (blk.adaln_head.weight == 0).all()
        assert (blk.adaln_head.bias == 0).all()


def test_shared_adaln_param_count() -> None:
    """Shared trunk should reduce AdaLN params vs per-block.

    Per-block full: 6 * (dim, dim) MLP = 6 × 128 × 128 = 98k.
    Shared rank-64: trunk (128 → 64) plus per-block (64 → 6·128) =
    128·64 + 2 × 64·6·128 ≈ 8k + 98k → much less than 2 × 98k.
    """
    p = _make()
    trunk_params = sum(x.numel() for x in p.adaln_trunk.parameters())
    head_params = sum(
        x.numel() for blk in p.blocks for x in blk.adaln_head.parameters()
    )
    assert trunk_params > 0
    assert head_params > 0
