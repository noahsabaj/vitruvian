# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Test-time adaptation of the JEPA predictor inside the plan loop (AdaJEPA).

AdaJEPA (Wang, Bounou, LeCun, Ren — arXiv:2606.32026) shows a frozen latent
world model degrades under test-time distribution shift, and that a single
self-supervised gradient step on each executed transition — *inside* the
closed MPC loop — substantially recovers planning success. This ports that
plan-execute-adapt-replan idea to Vitruvian, with two deliberate departures
that keep the **frozen-prior thesis** intact:

* We adapt **only the plastic predictor**. The DINOv3 encoder and the patch
  projector stay frozen, so the prediction target (a projected DINOv3
  embedding) is *fixed*. That also makes AdaJEPA's ``stop-gradient``
  anti-collapse stabilizer unnecessary here — the target is not being
  trained, so it cannot collapse. Their ablation supports predictor-only
  adaptation: "most of the needed correction lies in the predictor" for
  shape/dynamics shift (dynamics shift being our likely humanoid regime).
* The adaptation loss is our own training objective in the projected latent, on
  the transitions the agent just executed — :func:`~vitruvian.training.losses.prediction_loss`
  for an autoregressive predictor, or :func:`~vitruvian.training.losses.prefix_prediction_loss`
  for a Fast-LeWM prefix predictor (selected automatically via
  ``jepa.is_prefix_predictor``). We reuse the ``"emb"`` path (pre-projected
  embeddings), so the frozen projector/encoder are not even in the graph.

Usage (per episode, in an MPC/plan loop)::

    adapter = TestTimeAdapter(jepa, history_size=3)
    adapter.reset()                     # each episode starts from the pretrained predictor
    for macro in range(...):
        U = planner.plan(...)           # uses the current (possibly adapted) predictor
        # execute U; collect the tail transitions of the macro as aligned
        # (projected embedding z_t, action a_t) pairs — enough to fill one
        # ``history_size + num_preds`` window — and push them:
        for z_t, a_t in tail_transitions:
            adapter.push(z_t, a_t)
        adapter.step(n_steps=1)         # 1 GD step on the recent buffer; predictor updated in place

The planner shares the ``jepa`` reference, so in-place predictor updates take
effect on the next ``plan()`` with no extra wiring.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import torch

from vitruvian.training.losses import prediction_loss, prefix_prediction_loss


class TestTimeAdapter:
    """Closed-loop test-time adapter for a JEPA's predictor.

    Args:
        jepa: the world model. Only ``jepa.predictor`` is adapted; the
            backbone (frozen prior) and patch projector are left untouched.
        history_size: context length ``T_hist`` the predictor consumes
            (match the training loss / planner ``history_size``).
        num_preds: rollout horizon for the adaptation loss. ``1`` (default)
            is a pure 1-step teacher-forced update — AdaJEPA's "adapt on the
            observed transition"; ``>=2`` adds the k-step rollout term.
        lr: test-time learning rate (AdaJEPA's default is the training LR).
        buffer_size: capacity of the recent-transition buffer; the step uses
            the most recent ``history_size + num_preds`` entries.
    """

    # "Test" here means test-*time*, not a pytest case; skip collection.
    __test__ = False

    def __init__(
        self,
        jepa: Any,
        *,
        history_size: int,
        num_preds: int = 1,
        lr: float = 5e-4,
        buffer_size: int = 8,
    ) -> None:
        if getattr(jepa, "predictor", None) is None:
            raise ValueError("TestTimeAdapter requires a jepa with a predictor")
        self.jepa = jepa
        # Fast-LeWM always anchors on a single observed frame, so a prefix
        # predictor adapts with history_size=1 regardless of what the planner
        # uses for its rollout window.
        self.is_prefix = bool(getattr(jepa, "is_prefix_predictor", False))
        self.history_size = 1 if self.is_prefix else int(history_size)
        self.num_preds = max(1, int(num_preds))
        self.lr = float(lr)
        self.seq_len = self.history_size + self.num_preds
        self.buffer_size = max(int(buffer_size), self.seq_len)

        # Adapt ONLY the predictor's trainable params — never the frozen
        # DINOv3 backbone or the patch projector. This is the thesis-safety
        # boundary, enforced by construction (the optimizer never sees them).
        self.params = [p for p in jepa.predictor.parameters() if p.requires_grad]
        if not self.params:
            raise ValueError("predictor has no trainable parameters to adapt")
        self._pretrained = [p.detach().clone() for p in self.params]
        self.opt = torch.optim.Adam(self.params, lr=self.lr)

        self._emb: deque[torch.Tensor] = deque(maxlen=self.buffer_size)
        self._act: deque[torch.Tensor] = deque(maxlen=self.buffer_size)

    def reset(self) -> None:
        """Restore the pretrained predictor and clear the buffer. Call at the
        start of every episode — adaptation is per-episode (AdaJEPA keeps each
        episode's changes local)."""
        with torch.no_grad():
            for p, saved in zip(self.params, self._pretrained):
                p.copy_(saved)
        self.opt = torch.optim.Adam(self.params, lr=self.lr)
        self._emb.clear()
        self._act.clear()

    def push(self, emb: torch.Tensor, action: torch.Tensor) -> None:
        """Append one executed transition: the frame's *projected* embedding
        ``emb`` (``(N, D)`` patches or ``(D,)`` flat) and the action ``a``
        (``(action_dim,)``) taken at it. Both are detached."""
        self._emb.append(emb.detach())
        self._act.append(action.detach().float())

    def step(self, n_steps: int = 1) -> float | None:
        """Run ``n_steps`` self-supervised gradient steps on the most recent
        window. No-op (returns ``None``) until the buffer holds a full
        ``seq_len`` window. Returns the last loss value."""
        if len(self._emb) < self.seq_len:
            return None
        dev = next(self.jepa.predictor.parameters()).device
        emb = torch.stack(list(self._emb)[-self.seq_len :], dim=0).unsqueeze(0).to(dev)
        act = torch.stack(list(self._act)[-self.seq_len :], dim=0).unsqueeze(0).to(dev)
        batch = {"emb": emb, "action": act}
        rollout_weight = 1.0 if self.num_preds >= 2 else 0.0
        # Prefix predictors need the dense-prefix objective; the AR-only
        # prediction_loss would call model.predict() and hit the prefix
        # predictor's 3-D anchor assertion.
        loss_fn = prefix_prediction_loss if self.is_prefix else prediction_loss

        was_training = bool(getattr(self.jepa, "training", False))
        self.jepa.eval()  # deterministic adapt: no predictor dropout
        last: float | None = None
        for _ in range(int(n_steps)):
            self.opt.zero_grad(set_to_none=True)
            out = loss_fn(
                self.jepa,
                batch,
                history_size=self.history_size,
                num_preds=self.num_preds,
                rollout_weight=rollout_weight,
                reg_weight=0.0,
                proprio_dropout=0.0,
            )
            out["loss"].backward()
            self.opt.step()
            last = float(out["loss"].detach())
        if was_training:
            self.jepa.train()
        return last


__all__ = ["TestTimeAdapter"]
