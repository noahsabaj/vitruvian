# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""High-level MPPI planner for HWM-on-LeWM.

Plans a short sequence of macro-action latents (dim=macro_act_dim=32)
that drive the predicted CLS embedding toward a goal embedding. The
first planned latent is then decoded into primitive actions (via
``goal_builder.MacroNNRetriever``) and executed on the real
environment. Replanning occurs at macro boundaries.

Structure mirrors ``external/hwm/pldm/planning/planners/mppi_planner.py``
and the hierarchical wrapper in ``two_lvl_planner.py``, but adapted
to LeWM's flat CLS latent and G1's joint-torque action space.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone_adapter import LeWMBackboneAdapter
from .high_level import HighLevelModel
from .mppi import MPPI


class GoalL2Cost:
    """Running cost: ``||state − goal||²`` at each timestep.

    MPPI consumes ``sum_all_diffs`` / ``sum_last_n`` attrs to decide
    whether to sum across the whole rollout or only terminal steps.
    We use terminal-only (``sum_last_n=1``), matching HWM's maze
    config — the goal is a terminal-state specification, not a
    stage-wise reward shaping.
    """

    sum_all_diffs: bool = False
    sum_last_n: int = 1

    def __init__(self, goal_emb: torch.Tensor) -> None:
        self.goal_emb = goal_emb.detach().clone()

    def __call__(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # state: (K, D_latent), u: (K, macro_act_dim)
        diff = state - self.goal_emb.to(state.device, state.dtype)
        return diff.pow(2).sum(dim=-1)


class HighLevelPlanner(nn.Module):
    """MPPI over macro-action latents using the HL dynamics model.

    At plan time, we run MPPI with:
        state  ∈ R^{D_latent=192}    — current LeWM CLS embedding
        action ∈ R^{macro_act_dim=32} — macro-action latent
        dynamics(z, l) = HighLevelModel.predictor(z, l)
        running_cost = ||z − z_goal||² (terminal only)

    The HL predictor was trained as a transformer with causal masking
    over short histories; for MPPI we call it with sequence length 1
    at each step (the transformer's T=1 behaviour is well-defined: a
    single self-attention token produces the one-step-ahead delta).
    """

    def __init__(
        self,
        hl_model: HighLevelModel,
        goal_emb: torch.Tensor,
        *,
        horizon: int = 2,
        num_samples: int = 2000,
        noise_sigma: float = 10.0,
        lambda_: float = 0.0025,
        z_reg_coeff: float = 0.01,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.hl_model = hl_model.to(device).eval()
        self.device = device
        self.macro_act_dim = int(hl_model.action_encoder.macro_act_dim)
        self.latent_dim = int(hl_model.backbone.output_dim)
        self.horizon = int(horizon)

        if goal_emb.shape[-1] != self.latent_dim:
            raise ValueError(
                f"goal_emb last dim {goal_emb.shape[-1]} != "
                f"latent_dim {self.latent_dim}"
            )
        self.goal_emb = goal_emb.detach().to(device).view(self.latent_dim)

        predictor = self.hl_model.predictor

        def dynamics(state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            # MPPI's _expand_input_for_ensemble returns (M, K, D) where
            # M = rollout_samples (=1 for us). u has shape (K, nu).
            # We reshape to (B, 1, D/nu) to call the causal predictor
            # with sequence length 1, then restore the leading dims.
            squeezed = state.dim() == 3
            if squeezed:
                M, K, D = state.shape
                state_flat = state.reshape(M * K, D)
                if u.shape[0] == K and M > 1:
                    u_flat = u.repeat(M, 1)
                else:
                    u_flat = u
            else:
                state_flat = state
                u_flat = u
            with torch.no_grad():
                z = state_flat.unsqueeze(1)  # (B, 1, D)
                a = u_flat.unsqueeze(1)      # (B, 1, nu)
                nxt = predictor(z, a).squeeze(1)  # (B, D)
            if squeezed:
                return nxt.reshape(M, K, D)
            return nxt

        self.cost_fn = GoalL2Cost(self.goal_emb)
        noise_sigma_t = torch.eye(self.macro_act_dim, device=device) * noise_sigma
        self.mppi = MPPI(
            dynamics=dynamics,
            running_cost=self.cost_fn,
            nx=self.latent_dim,
            noise_sigma=noise_sigma_t,
            num_samples=num_samples,
            horizon=self.horizon,
            lambda_=lambda_,
            device=device,
            u_per_command=self.horizon,  # return full plan
            latent_actions=True,
            z_reg_coeff=z_reg_coeff,
        )

    @torch.no_grad()
    def plan(
        self, current_emb: torch.Tensor, shift_nominal: bool = True
    ) -> torch.Tensor:
        """Plan a macro-action latent sequence from the current latent.

        Args:
            current_emb: (D_latent,) or (1, 1, D_latent) — current CLS.
            shift_nominal: whether to shift the warm-started nominal
                trajectory forward before re-planning.
        Returns:
            U: (horizon, macro_act_dim) planned macro latents.
        """
        emb = current_emb.detach().to(self.device).view(self.latent_dim)
        plan = self.mppi.command(emb, shift_nominal_trajectory=shift_nominal)
        # u_per_command = horizon makes command() return the full U.
        return plan.detach()

    def reset(self) -> None:
        """Clear nominal trajectory between episodes."""
        self.mppi.reset()

    def set_goal(self, goal_emb: torch.Tensor) -> None:
        """Update the target embedding without rebuilding the planner."""
        self.goal_emb = goal_emb.detach().to(self.device).view(self.latent_dim)
        self.cost_fn.goal_emb = self.goal_emb.clone()

    @torch.no_grad()
    def predict_subgoal(
        self, current_emb: torch.Tensor, macro_plan: torch.Tensor
    ) -> torch.Tensor:
        """Given the HL plan, roll the HL predictor forward one macro and
        return the first predicted waypoint — this is the L1 subgoal.

        This mirrors HWM's ``TwoLvlPlanner.plan``: ``l2_result.pred_obs[1]``
        (external/hwm/pldm/planning/planners/two_lvl_planner.py:74). The
        HL predictor is autoregressive over macros; we only need the
        first prediction since L1 will replan every macro boundary.

        Args:
            current_emb: (D_latent,) current CLS.
            macro_plan: (horizon, macro_act_dim) planned macro latents.
        Returns:
            subgoal: (D_latent,) first-macro predicted waypoint.
        """
        z = current_emb.view(1, 1, self.latent_dim).to(self.device)
        # The HL action encoder trained on primitive-action chunks but
        # also needs a latent-macro input for planning-time rollouts. For
        # ``predict_subgoal`` we are using latent macros directly
        # (already encoded by the planner's action space), so wrap them
        # to match predictor's expected shape: (B, T, macro_act_dim).
        a = macro_plan[:1].to(self.device).view(1, 1, self.macro_act_dim)
        nxt = self.hl_model.predictor(z, a).squeeze(0).squeeze(0)
        return nxt  # (D_latent,)


class LowLevelPlanner(nn.Module):
    """MPPI over primitive actions using LeWM's predictor as dynamics.

    Mirrors HWM's L1 planner in
    ``external/hwm/pldm/planning/planners/mppi_planner.py``, adapted
    for LeWM's autoregressive JEPA: we use ``jepa.rollout(info,
    action_sequence, history_size)`` which natively batches over
    candidate plans.

    CEM-style update: sample K perturbed primitive sequences, roll each
    out through LeWM, score by terminal L2 distance to ``subgoal_emb``,
    softmax-weight the perturbations, and update the nominal plan.
    """

    def __init__(
        self,
        lewm_jepa: nn.Module,
        backbone: LeWMBackboneAdapter,
        subgoal_emb: torch.Tensor,
        *,
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
        self.jepa = lewm_jepa
        self.backbone = backbone
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.noise_sigma = noise_sigma
        self.lambda_ = lambda_
        self.iterations = iterations
        self.history_size = history_size
        self.action_low = action_low
        self.action_high = action_high
        self.device = device

        if subgoal_emb.shape[-1] != backbone.output_dim:
            raise ValueError(
                f"subgoal_emb last dim {subgoal_emb.shape[-1]} != "
                f"backbone output dim {backbone.output_dim}"
            )
        self.subgoal_emb = subgoal_emb.detach().to(device).view(-1)

        # Warm-started nominal trajectory for receding-horizon reuse.
        self.U: torch.Tensor | None = None

    def set_subgoal(self, subgoal_emb: torch.Tensor) -> None:
        self.subgoal_emb = subgoal_emb.detach().to(self.device).view(-1)

    def reset(self) -> None:
        self.U = None

    @torch.no_grad()
    def plan(
        self,
        pixel_history: torch.Tensor,
        action_history: torch.Tensor,
        warm_start_U: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Plan ``horizon`` primitive actions via CEM-MPPI against subgoal.

        Args:
            pixel_history: (H_hist, 3, H, W) last-``history_size`` head-cam
                frames in [0, 1] float or uint8. If fewer than
                ``history_size`` provided, we pad by repeating the earliest.
            action_history: (H_hist, action_dim) matching primitive actions.
                Zero-pad if absent.
            warm_start_U: (horizon, action_dim) optional nominal plan to
                initialize from — e.g. an expert PPO policy rolled out
                from the current env state. When provided, overrides both
                zero-init and the shifted self.U, because an expert prior
                in the current state is more informative than yesterday's
                refined plan. This is the PLDM "expert-prior nominal"
                trick — puts MPPI's Gaussian sample cloud inside the
                world-model training distribution.
        Returns:
            U: (horizon, action_dim) planned primitive actions.
        """
        device = self.device
        HS = self.history_size
        H = self.horizon
        K = self.num_samples
        A = self.action_dim

        # Left-pad the history to exactly HS frames.
        ph = pixel_history.to(device)
        if ph.dtype == torch.uint8:
            ph = ph.float() / 255.0
        ah = action_history.to(device).float()
        if ph.shape[0] < HS:
            pad_n = HS - ph.shape[0]
            ph = torch.cat([ph[:1].expand(pad_n, -1, -1, -1), ph], dim=0)
            ah = torch.cat([torch.zeros(pad_n, A, device=device), ah], dim=0)
        ph = ph[-HS:]
        ah = ah[-HS:]

        # Shape everything to (B=1, S=K, T, ...) expected by rollout.
        pixels_KS = ph.unsqueeze(0).unsqueeze(0).expand(1, K, -1, -1, -1, -1)
        hist_acts_KS = ah.unsqueeze(0).unsqueeze(0).expand(1, K, -1, -1)

        # Initialize nominal plan. Priority: warm-start > shifted > zeros.
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

        for _ in range(self.iterations):
            noise = torch.randn(K, H, A, device=device) * self.noise_sigma
            candidates = (U.unsqueeze(0) + noise).clamp(
                self.action_low, self.action_high
            )  # (K, H, A)

            future_KS = candidates.unsqueeze(0)  # (1, K, H, A)
            action_seq = torch.cat([hist_acts_KS, future_KS], dim=2)
            # rollout mutates its `info` dict — build a fresh one each call.
            info = {"pixels": pixels_KS.clone()}
            out = self.jepa.rollout(info, action_seq, history_size=HS)

            pred_emb = out["predicted_emb"]  # (B, S, T_hist + horizon + 1, D)
            # The last index is the terminal prediction.
            pred_final = pred_emb[0, :, -1, :]  # (K, D)

            costs = ((pred_final - self.subgoal_emb) ** 2).sum(dim=-1)  # (K,)

            beta = costs.min()
            weights = torch.exp(-(costs - beta) / self.lambda_)
            weights = weights / (weights.sum() + 1e-12)

            # Weighted update on the NOMINAL plan (not on the samples) —
            # standard MPPI perturbation update.
            delta = (weights.view(K, 1, 1) * noise).sum(dim=0)  # (H, A)
            U = (U + delta).clamp(self.action_low, self.action_high)

            if beta < best_cost:
                best_cost = beta
                best_U = candidates[costs.argmin()].clone()

        self.U = U
        self.best_cost = float(best_cost)
        self.best_U = best_U
        return U


class HierarchicalPlanner:
    """Two-level MPPI: L2 over macro-latents, L1 over primitive actions.

    Structure mirrors ``external/hwm/pldm/planning/planners/two_lvl_planner.py``.
    Use as:

        hp = HierarchicalPlanner(hl_planner, lewm_jepa, backbone, horizon_primitives=50)
        hp.set_goal(goal_emb)
        for macro_idx in range(n_macros):
            primitives = hp.plan_primitives(pixel_history, action_history)
            for p in primitives:
                env.step(p); update history
    """

    def __init__(
        self,
        hl_planner: HighLevelPlanner,
        lewm_jepa: nn.Module,
        backbone: LeWMBackboneAdapter,
        *,
        horizon_primitives: int = 50,
        num_samples_l1: int = 500,
        noise_sigma_l1: float = 0.3,
        lambda_l1: float = 0.0025,
        iterations_l1: int = 3,
        history_size: int = 3,
        device: str = "cuda",
    ) -> None:
        self.hl_planner = hl_planner
        self.jepa = lewm_jepa
        self.backbone = backbone
        self.device = device

        # Pre-build the L1 planner with a placeholder subgoal; we update
        # it every macro via `set_subgoal`.
        dummy = torch.zeros(backbone.output_dim, device=device)
        self.l1_planner = LowLevelPlanner(
            lewm_jepa=lewm_jepa,
            backbone=backbone,
            subgoal_emb=dummy,
            horizon=horizon_primitives,
            num_samples=num_samples_l1,
            noise_sigma=noise_sigma_l1,
            lambda_=lambda_l1,
            iterations=iterations_l1,
            history_size=history_size,
            device=device,
        )

    def set_goal(self, goal_emb: torch.Tensor) -> None:
        self.hl_planner.set_goal(goal_emb)

    def reset(self) -> None:
        self.hl_planner.reset()
        self.l1_planner.reset()

    @torch.no_grad()
    def plan_primitives(
        self,
        current_emb: torch.Tensor,
        pixel_history: torch.Tensor,
        action_history: torch.Tensor,
        shift_nominal: bool = True,
        warm_start_U: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Two-level plan → ``horizon_primitives`` primitive actions.

        Args:
            current_emb: (D_latent,) current CLS embedding.
            pixel_history: (H_hist, 3, H, W) most recent head-cam frames.
            action_history: (H_hist, action_dim) primitives executed so far.
            shift_nominal: whether the HL planner shifts its warm start.
            warm_start_U: (horizon_primitives, action_dim) optional expert
                prior (e.g. PPO-policy rollout) for the L1 MPPI nominal.

        Returns:
            primitives: (horizon_primitives, action_dim) planned primitives.
            info: dict with planning diagnostics (subgoal emb, macro plan).
        """
        macro_plan = self.hl_planner.plan(current_emb, shift_nominal=shift_nominal)
        subgoal = self.hl_planner.predict_subgoal(current_emb, macro_plan)
        self.l1_planner.set_subgoal(subgoal)

        primitives = self.l1_planner.plan(
            pixel_history, action_history, warm_start_U=warm_start_U
        )
        diag = {
            "macro_plan": macro_plan.detach().cpu(),
            "subgoal": subgoal.detach().cpu(),
            "l1_best_cost": float(self.l1_planner.best_cost),
        }
        return primitives, diag


def encode_goal(
    backbone: LeWMBackboneAdapter, goal_pixel: torch.Tensor
) -> torch.Tensor:
    """Encode a single goal image to a flat CLS embedding.

    Accepts (H, W, 3) uint8, (3, H, W) uint8/float, or (B, 3, H, W)
    float in [0, 1]. Returns a 1-D tensor of shape (D_latent,).
    """
    if goal_pixel.dim() == 3 and goal_pixel.shape[-1] == 3:
        goal_pixel = goal_pixel.permute(2, 0, 1)  # → (3, H, W)
    if goal_pixel.dim() == 3:
        goal_pixel = goal_pixel.unsqueeze(0)      # (1, 3, H, W)
    if goal_pixel.dtype == torch.uint8:
        goal_pixel = goal_pixel.float() / 255.0
    goal_pixel = goal_pixel.unsqueeze(1)          # (B, 1, 3, H, W)
    emb = backbone.encode(goal_pixel)             # (B, 1, D)
    return emb.squeeze(0).squeeze(0)              # (D,)
