# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""VF-HER IQL trainer — expectile asymmetry, EMA, terminal bootstrap drop."""

from __future__ import annotations

import torch
import torch.nn as nn

from vitruvian.training import (
    VFHERConfig,
    VFHERTrainer,
    ValueHead,
    ema_update,
    expectile_loss,
)


def test_expectile_loss_asymmetry() -> None:
    # tau=0.7 up-weights positive residuals 0.7 vs negative 0.3.
    pos = expectile_loss(torch.tensor([1.0]), expectile=0.7)
    neg = expectile_loss(torch.tensor([-1.0]), expectile=0.7)
    assert torch.allclose(pos, torch.tensor(0.7), atol=1e-6)
    assert torch.allclose(neg, torch.tensor(0.3), atol=1e-6)


def test_ema_update_moves_target_toward_source() -> None:
    tgt = nn.Linear(4, 4)
    src = nn.Linear(4, 4)
    with torch.no_grad():
        for p in tgt.parameters():
            p.zero_()
        for p in src.parameters():
            p.fill_(1.0)
    ema_update(tgt, src, rate=0.1)
    # target := 0.9*0 + 0.1*1 == 0.1
    for p in tgt.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.1), atol=1e-6)


def _batch(B: int, D: int, terminal: bool) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    flag = torch.ones(B) if terminal else torch.zeros(B)
    return {
        "emb_t": torch.randn(B, D), "prop_t": torch.randn(B, 4),
        "emb_tp1": torch.randn(B, D), "prop_tp1": torch.randn(B, 4),
        "emb_g": torch.randn(B, D), "prop_g": torch.randn(B, 4),
        "is_goal_reached": flag.bool(),
    }


def test_terminal_transitions_drop_the_bootstrap() -> None:
    """At goal-reached (terminal) transitions the trainer must zero the
    next-state bootstrap, so the target is exactly ``reward_self_loop`` (0). If
    the bootstrap leaked in, every transition would bootstrap and V would
    collapse to the constant ``reward_step / (1 - gamma)`` — the bug the
    terminal mask fixes."""
    D = 16
    trainer = VFHERTrainer(
        ValueHead(emb_dim=D, hidden=32, out_dim=8), nn.Linear(4, D), VFHERConfig()
    )
    term = trainer.compute_loss(_batch(8, D, terminal=True))
    nonterm = trainer.compute_loss(_batch(8, D, terminal=False))

    # Terminal: v_next zeroed → target == reward_self_loop (0.0).
    assert abs(float(term["target_mean"]) - VFHERConfig().reward_self_loop) < 1e-5
    # Non-terminal: target = reward_step + gamma * v_next (v_next < 0) → negative
    # and materially different from the terminal target.
    assert float(nonterm["target_mean"]) < -0.5
