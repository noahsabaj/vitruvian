# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""MPPI primitive-action planner.

Ports ``LowLevelPlanner`` from the legacy ``vitruvian.hwm`` package,
generalizes the cost computation to a pluggable strategy
(:class:`vitruvian.planning.costs.CostFn`), and deletes the MC-dropout
uncertainty path (M4.3 research code that never paid off and was kept
gated at defaults only).

The planner runs CEM-style MPPI over primitive actions:

    1. Sample K perturbed sequences around a nominal plan ``U``.
    2. Roll each through the JEPA world model.
    3. Score by the supplied cost strategy.
    4. Softmax-reweight perturbations and update ``U``.

Supports two input modes (same as legacy):

* **Pre-encoded (preferred)**: caller passes an ``encoded_history``
  tensor — no DINOv3 forward at plan time.
* **Pixel (legacy)**: caller passes raw pixels — JEPA re-encodes.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vitruvian.planning.costs import CostFn, MSECost


class MPPIPlanner(nn.Module):
    """MPPI planner over primitive joint-torque actions."""

    def __init__(
        self,
        jepa: nn.Module,
        backbone: nn.Module,
        subgoal_emb: torch.Tensor,
        *,
        cost_fn: CostFn | None = None,
        action_dim: int = 29,
        horizon: int = 50,
        num_samples: int = 500,
        noise_sigma: float = 0.3,
        lambda_: float = 0.0025,
        iterations: int = 3,
        history_size: int = 3,
        action_low: float = -1.0,
        action_high: float = 1.0,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        # Any: JEPA composer exposes .rollout(info, ...) with a dynamic
        # dict contract; backbone exposes .encode() and .output_dim as
        # duck-typed attrs. Both cases are Tensor-vs-Module narrowed by
        # nn.Module's dynamic-attr access unless we skip typing them.
        self.jepa: Any = jepa
        self.backbone = backbone
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.noise_sigma = float(noise_sigma)
        self.lambda_ = float(lambda_)
        self.iterations = int(iterations)
        self.history_size = int(history_size)
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.device = device

        if subgoal_emb.shape[-1] != backbone.output_dim:
            raise ValueError(
                f"subgoal_emb last dim {subgoal_emb.shape[-1]} != "
                f"backbone output dim {backbone.output_dim}"
            )
        # Preserve full shape — ``(D,)`` for CLS encoders, ``(N, D)`` for
        # patch encoders. Cost strategies handle broadcasting.
        self.subgoal_emb = subgoal_emb.detach().to(device)

        self.cost_fn: CostFn = cost_fn if cost_fn is not None else MSECost()

        # Warm-started nominal trajectory for receding-horizon reuse.
        self.U: torch.Tensor | None = None

        # Populated after every ``plan()`` call.
        self.best_cost: float = float("nan")
        self.best_U: torch.Tensor | None = None
        self.best_c_goal: float = float("nan")
        self.c_goal_min: float = float("nan")
        self.c_goal_max: float = float("nan")

    def set_subgoal(self, subgoal_emb: torch.Tensor) -> None:
        self.subgoal_emb = subgoal_emb.detach().to(self.device)

    def reset(self) -> None:
        self.U = None

    @torch.no_grad()
    def plan(
        self,
        pixel_history: torch.Tensor | None,
        action_history: torch.Tensor,
        warm_start_U: torch.Tensor | None = None,
        *,
        encoded_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Plan ``horizon`` primitive actions via CEM-MPPI.

        Exactly one of ``pixel_history`` or ``encoded_history`` must be
        provided. Pre-encoded mode (``encoded_history``) skips all
        per-iteration encoder calls by passing a pre-populated
        ``"emb"`` field straight into ``jepa.rollout``.

        Args:
            pixel_history: ``(H_hist, 3, H, W)`` head-cam frames, uint8
                or float in ``[0, 1]``.
            action_history: ``(H_hist, action_dim)`` primitive actions.
            warm_start_U: ``(horizon, action_dim)`` optional PPO-policy
                rollout to seed MPPI's Gaussian sample cloud.
            encoded_history: ``(H_hist, *emb_shape)`` pre-encoded frame
                history. Mutually exclusive with ``pixel_history``.
        Returns:
            ``U (horizon, action_dim)`` planned primitive actions.
        """
        if (pixel_history is None) == (encoded_history is None):
            raise ValueError(
                "plan() requires exactly one of pixel_history or "
                "encoded_history; got "
                f"{'both' if pixel_history is not None else 'neither'}."
            )
        device = self.device
        HS = self.history_size
        H = self.horizon
        K = self.num_samples
        A = self.action_dim

        ah = action_history.to(device).float()

        pixels_KS: torch.Tensor | None = None
        emb_KS: torch.Tensor | None = None
        if encoded_history is not None:
            eh = encoded_history.to(device).float()
            if eh.shape[0] < HS:
                pad_n = HS - eh.shape[0]
                pad_tile = eh[:1].expand((pad_n,) + tuple(eh.shape[1:]))
                eh = torch.cat([pad_tile, eh], dim=0)
                ah = torch.cat([torch.zeros(pad_n, A, device=device), ah], dim=0)
            eh = eh[-HS:]
            emb_KS = (
                eh.unsqueeze(0)
                .unsqueeze(0)
                .expand((1, K) + tuple(eh.shape))
                .contiguous()
            )
        else:
            assert pixel_history is not None
            ph = pixel_history.to(device)
            if ph.dtype == torch.uint8:
                ph = ph.float() / 255.0
            if ph.shape[0] < HS:
                pad_n = HS - ph.shape[0]
                ph = torch.cat([ph[:1].expand(pad_n, -1, -1, -1), ph], dim=0)
                ah = torch.cat([torch.zeros(pad_n, A, device=device), ah], dim=0)
            ph = ph[-HS:]
            pixels_KS = ph.unsqueeze(0).unsqueeze(0).expand(
                1, K, -1, -1, -1, -1
            )

        ah = ah[-HS:]
        hist_acts_KS = ah.unsqueeze(0).unsqueeze(0).expand(1, K, -1, -1)

        # Priority: warm-start > shifted (receding-horizon) > zeros.
        if warm_start_U is not None:
            U = warm_start_U.to(device).float().clamp(
                self.action_low, self.action_high
            )
            if U.shape != (H, A):
                raise ValueError(
                    f"warm_start_U shape {tuple(U.shape)} != ({H}, {A})"
                )
        elif self.U is None:
            U = torch.zeros(H, A, device=device)
        else:
            U = torch.roll(self.U, shifts=-1, dims=0)
            U[-1] = 0.0

        best_U = U.clone()
        best_cost = torch.tensor(float("inf"), device=device)
        last_c_goal: torch.Tensor | None = None

        for _ in range(self.iterations):
            noise = torch.randn(K, H, A, device=device) * self.noise_sigma
            candidates = (U.unsqueeze(0) + noise).clamp(
                self.action_low, self.action_high
            )

            future_KS = candidates.unsqueeze(0)  # (1, K, H, A)
            action_seq = torch.cat([hist_acts_KS, future_KS], dim=2)

            if emb_KS is not None:
                info = {"emb": emb_KS.clone()}
            else:
                assert pixels_KS is not None
                info = {"pixels": pixels_KS.clone()}
            out = self.jepa.rollout(info, action_seq, history_size=HS)
            pred_emb = out["predicted_emb"]  # (1, K, T_total, *latent)
            pred_final = pred_emb[0, :, -1]   # (K, *latent)
            costs = self.cost_fn(pred_final, self.subgoal_emb)  # (K,)
            last_c_goal = costs

            beta = costs.min()
            weights = torch.exp(-(costs - beta) / self.lambda_)
            weights = weights / (weights.sum() + 1e-12)

            delta = (weights.view(K, 1, 1) * noise).sum(dim=0)  # (H, A)
            U = (U + delta).clamp(self.action_low, self.action_high)

            if beta < best_cost:
                best_cost = beta
                best_U = candidates[costs.argmin()].clone()

        self.U = U
        self.best_cost = float(best_cost)
        self.best_U = best_U
        if last_c_goal is not None:
            self.best_c_goal = float(last_c_goal.min())
            self.c_goal_min = float(last_c_goal.min())
            self.c_goal_max = float(last_c_goal.max())
        return U


def encode_goal(backbone: Any, goal_pixel: torch.Tensor) -> torch.Tensor:
    """Encode a single goal image through any :class:`Backbone`.

    ``backbone`` is duck-typed to the :class:`~vitruvian.models.Backbone`
    protocol (``.encode(pixels)`` and ``.output_dim``). Accepts
    ``(H, W, 3)`` uint8, ``(3, H, W)`` uint8/float, or
    ``(B, 3, H, W)`` float in ``[0, 1]``. Returns the backbone's native
    per-frame latent (trailing dims preserved, leading batch + time
    dims squeezed).
    """
    if goal_pixel.dim() == 3 and goal_pixel.shape[-1] == 3:
        goal_pixel = goal_pixel.permute(2, 0, 1)
    if goal_pixel.dim() == 3:
        goal_pixel = goal_pixel.unsqueeze(0)
    if goal_pixel.dtype == torch.uint8:
        goal_pixel = goal_pixel.float() / 255.0
    goal_pixel = goal_pixel.unsqueeze(1)
    emb: torch.Tensor = backbone.encode(goal_pixel)
    return emb.squeeze(0).squeeze(0)


__all__ = ["MPPIPlanner", "encode_goal"]
