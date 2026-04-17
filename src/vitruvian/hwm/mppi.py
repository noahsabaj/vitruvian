# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT  (upstream UM-ARM-Lab/pytorch_mppi)
#
# Vendored from external/hwm/pldm/planning/planners/mppi_torch.py
# with PLDM-specific additions stripped:
#
#   - removed `proprio_dim`, `proprio`, `location`, `raw_location`
#     params and state (PLDM backbone decomposition — not applicable
#     to LeWM's flat CLS latent; see ADR 008)
#   - removed `cost_entity` switch and the `ensemble_{obs,proprio,
#     location}_component` reads; our dynamics returns a single
#     next-state tensor
#   - removed `rollout_obs_var_cost`, `rollout_proprio_var_cost`
#     (kept `rollout_var_cost` — applies to the single state tensor)
#
# The core Williams et al. 2017 MPPI algorithm (algorithm 2) is
# preserved verbatim: noise sampling, cost evaluation over sampled
# trajectories, softmax importance weighting, nominal trajectory
# update. Upstream repo:
# https://github.com/UM-ARM-Lab/pytorch_mppi (MIT).

import logging

import torch
from torch.distributions.multivariate_normal import MultivariateNormal

logger = logging.getLogger(__name__)


def _ensure_non_zero(cost, beta, factor):
    return torch.exp(-factor * (cost - beta))


class MPPI:
    """Model Predictive Path Integral control.

    Batch-samples trajectories so it scales well with the number of
    samples K. Per Williams et al. 2017,
    'Information Theoretic MPC for Model-Based Reinforcement Learning'.

    Dynamics signature:
        dynamics(state, u) -> next_state
        state: (K, nx), u: (K, nu) -> (K, nx) or (M, K, nx) with ensembles.

    Running cost signature:
        running_cost(state, u) -> (K,) cost per trajectory.
        Optional attributes read: `sum_all_diffs: bool`,
        `sum_last_n: Optional[int]`.
    """

    def __init__(
        self,
        dynamics,
        running_cost,
        nx,
        noise_sigma,
        num_samples=100,
        horizon=15,
        device="cpu",
        terminal_state_cost=None,
        lambda_=1.0,
        noise_mu=None,
        action_normalizer=None,
        u_init=None,
        U_init=None,
        u_scale=1,
        u_per_command=1,
        latent_actions=False,
        z_reg_coeff=0.1,
        step_dependent_dynamics=False,
        rollout_samples=1,
        var_samples=0,
        rollout_var_cost=0,
        rollout_var_discount=0.95,
        sample_null_action=False,
        noise_abs_cost=False,
    ):
        self.d = device
        self.dtype = noise_sigma.dtype
        self.K = num_samples
        self.T = horizon
        self.latent_actions = latent_actions
        self.z_reg_coeff = z_reg_coeff

        self.nx = nx
        self.nu = 1 if len(noise_sigma.shape) == 0 else noise_sigma.shape[0]
        self.lambda_ = lambda_

        if noise_mu is None:
            noise_mu = torch.zeros(self.nu, dtype=self.dtype)
        if u_init is None:
            u_init = torch.zeros_like(noise_mu)
        if self.nu == 1:
            noise_mu = noise_mu.view(-1)
            noise_sigma = noise_sigma.view(-1, 1)

        self.action_normalizer = action_normalizer
        self.u_scale = u_scale
        self.u_per_command = u_per_command

        self.noise_mu = noise_mu.to(self.d)
        self.noise_sigma = noise_sigma.to(self.d)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(
            self.noise_mu, covariance_matrix=self.noise_sigma
        )
        self.U = U_init
        self.u_init = u_init.to(self.d)
        if self.U is None:
            self.U = self.noise_dist.sample((self.T,))

        self.step_dependency = step_dependent_dynamics
        self.F = dynamics
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost
        self.sample_null_action = sample_null_action
        self.noise_abs_cost = noise_abs_cost
        self.state = None

        # Ensemble / variance handling
        self.M = rollout_samples
        self.var_samples = var_samples
        self.rollout_var_cost = rollout_var_cost
        self.rollout_var_discount = rollout_var_discount
        if self.var_samples:
            assert self.M == 1

        # sampled results from last command
        self.cost_total = None
        self.cost_total_non_zero = None
        self.omega = None
        self.states = None
        self.actions = None

    def _dynamics(self, state, u):
        return self.F(state, u)

    def _running_cost(self, state, u, t):
        return (
            self.running_cost(state, u, t)
            if self.step_dependency
            else self.running_cost(state, u)
        )

    def _running_var_cost(self, states):
        """Variance of features across ensemble rollouts → (K,)."""
        var = torch.var(states, dim=0, unbiased=False)
        return var.sum(dim=-1)

    def shift_nominal_trajectory(self):
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = self.u_init

    def command(self, state, shift_nominal_trajectory=True):
        """Return the best action (``(nu,)``) given current state.

        Args:
            state: (nx,) or (K, nx). A K×nx tensor may be used to
                propagate a distribution of initial states.
            shift_nominal_trajectory: shift forward before replanning.
        """
        if shift_nominal_trajectory:
            self.shift_nominal_trajectory()
        return self._command(state)

    def _command(self, state):
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype, device=self.d)
        cost_total = self._compute_total_cost_batch()
        beta = torch.min(cost_total)
        self.cost_total_non_zero = _ensure_non_zero(cost_total, beta, 1 / self.lambda_)
        eta = torch.sum(self.cost_total_non_zero)
        self.omega = (1.0 / eta) * self.cost_total_non_zero

        perturbations = []
        for t in range(self.T):
            perturbations.append(
                torch.sum(self.omega.view(-1, 1) * self.noise[:, t], dim=0)
            )
        perturbations = torch.stack(perturbations)
        self.U = self.U + perturbations

        if self.u_per_command == -1:
            return self.U
        action = self.U[: self.u_per_command]
        if self.u_per_command == 1:
            action = action[0]
        return action

    def change_horizon(self, horizon):
        if horizon < self.U.shape[0]:
            self.U = self.U[:horizon]
        elif horizon > self.U.shape[0]:
            self.U = torch.cat(
                (self.U, self.u_init.repeat(horizon - self.U.shape[0], 1))
            )
        self.T = horizon

    def reset(self):
        """Clear controller state after a trial."""
        self.U = self.noise_dist.sample((self.T,))

    def _expand_input_for_ensemble(self, x, K, state_dim):
        if x is None:
            return None
        if x.shape != (K, state_dim):
            x = x.repeat(K, *[1 for _ in x.shape])
        return x.repeat(self.M, *[1 for _ in x.shape])

    def _compute_rollout_costs(self, perturbed_actions):
        K, T, nu = perturbed_actions.shape
        assert nu == self.nu

        cost_total = torch.zeros(K, device=self.d, dtype=self.dtype)
        cost_samples = cost_total.repeat(self.M, 1)
        cost_var = torch.zeros_like(cost_total)

        state = self._expand_input_for_ensemble(self.state, K=K, state_dim=self.nx)

        actions = []
        # sum_all_diffs / sum_last_n are optional attrs on running_cost
        # allowing the cost function to only accumulate terminal or
        # last-n timesteps. Defaults: sum everything.
        sum_all_diffs = getattr(self.running_cost, "sum_all_diffs", True)
        sum_last_n = getattr(self.running_cost, "sum_last_n", None)
        idx_start = (
            0
            if sum_all_diffs
            else (T - sum_last_n if sum_last_n is not None else 0)
        )

        for t in range(T):
            u = self.u_scale * perturbed_actions[:, t]
            state = self._dynamics(state, u)
            c = self._running_cost(state, u, t)
            if t >= idx_start:
                cost_samples = cost_samples + c
            if self.M > 1 and self.rollout_var_cost:
                cost_var += self._running_var_cost(state) * (
                    self.rollout_var_discount**t
                )
            actions.append(u)

        actions = torch.stack(actions, dim=-2)
        states = None  # we don't store trajectories by default

        if self.terminal_state_cost:
            c = self.terminal_state_cost(states, actions)
            cost_samples = cost_samples + c

        if self.latent_actions:
            # regularize towards a standard-normal prior on latent actions
            prior_mus = torch.zeros_like(actions)
            prior_vars = torch.ones_like(actions)
            prior_d = torch.distributions.Normal(loc=prior_mus, scale=prior_vars)
            z_reg = -prior_d.log_prob(actions).mean(dim=(1, 2))
            z_reg = z_reg * self.z_reg_coeff
            cost_samples = cost_samples + z_reg

        cost_total = cost_total + cost_samples.mean(dim=0)
        cost_total = cost_total + cost_var * self.rollout_var_cost
        return cost_total, states, actions

    def _compute_total_cost_batch(self):
        noise = self.noise_dist.rsample((self.K, self.T))
        perturbed_action = self.U + noise
        if self.sample_null_action:
            perturbed_action[self.K - 1] = 0
        self.perturbed_action = self._bound_action(perturbed_action)
        self.noise = self.perturbed_action - self.U

        if self.noise_abs_cost:
            action_cost = self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv
        else:
            action_cost = self.lambda_ * self.noise @ self.noise_sigma_inv

        rollout_cost, self.states, actions = self._compute_rollout_costs(
            self.perturbed_action
        )
        self.actions = actions / self.u_scale
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2))
        self.cost_total = rollout_cost + perturbation_cost
        return self.cost_total

    def _bound_action(self, action):
        if self.action_normalizer is not None:
            for t in range(self.T):
                u = action[:, self._slice_control(t)]
                cu = self.action_normalizer(u)
                action[:, self._slice_control(t)] = cu
        return action

    def _slice_control(self, t):
        return slice(t * self.nu, (t + 1) * self.nu)
