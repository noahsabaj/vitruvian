# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""G1 MuJoCo playground env + PPO warm-start rollout helpers.

Builds ``G1JoystickFlatTerrain`` with head-cam, chase-cam, side-cam
attached; optionally loads a frozen PPO policy whose rollout provides a
warm-start nominal plan for MPPI.

All helpers are pure Python + NumPy (no torch), so they cost nothing to
import inside a CLI entrypoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from vitruvian.env.cameras import (
    CHASE_CAM_POS,
    CHASE_CAM_TARGET,
    HEAD_CAM_POS,
    HEAD_CAM_QUAT,
    SIDE_CAM_POS,
    look_at_quat,
)

ENV_NAME = "G1JoystickFlatTerrain"
ENV_OVERRIDES = {"njmax": 96}


def _resolve_g1_scene(repo_root: Path) -> Path:
    """Default path: ``external/mujoco_menagerie/unitree_g1/scene.xml``."""
    return repo_root / "external" / "mujoco_menagerie" / "unitree_g1" / "scene.xml"


def install_jax_brax_shim() -> None:
    """Monkey-patch ``jax.device_put_replicated`` for brax 0.14.x × JAX 0.10.

    brax 0.14 still calls ``jax.device_put_replicated`` which JAX 0.10
    removed. Replicate ourselves via ``tree.map``. Safe to call more
    than once — the shim is idempotent — and a no-op on any JAX version
    that still ships ``device_put_replicated`` (we never shadow the real
    implementation).
    """
    import jax
    import jax.numpy as jnp

    marker = "_vitruvian_replicate_shim"
    if getattr(jax, marker, False):
        return
    if hasattr(jax, "device_put_replicated"):
        # This JAX still provides it — use the real one, don't shadow it.
        setattr(jax, marker, True)
        return

    def _replicate(value, devices):
        try:
            n = len(devices)
        except TypeError:
            n = 1

        def _leaf(x):
            arr = jnp.asarray(x)
            expanded = jnp.expand_dims(arr, 0)
            if n > 1:
                expanded = jnp.broadcast_to(expanded, (n,) + arr.shape)
            return jax.device_put(expanded)

        return jax.tree.map(_leaf, value)

    jax.device_put_replicated = _replicate  # type: ignore[attr-defined]
    setattr(jax, marker, True)


def build_env_and_policy(
    ckpt_policy: Path | None,
    device: str,
    seed: int,
    *,
    repo_root: Path,
    env_name: str = ENV_NAME,
    env_overrides: dict[str, Any] | None = None,
) -> dict:
    """Build the G1 playground env with head/chase/side cameras and
    (optionally) a PPO policy whose rollout seeds MPPI's warm start.

    Returns a context dict with ``env``, ``state``, ``reset_fn``,
    ``step_fn``, ``mj_model``, ``mj_data``, ``cam_id``, ``cam_ids``,
    ``renderer``, ``rng``, ``policy``.
    """
    install_jax_brax_shim()

    import functools
    import pickle

    import jax
    import mujoco
    from brax.training import checkpoint as brax_checkpoint
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import registry
    from mujoco_playground.config import locomotion_params

    env_overrides = dict(env_overrides) if env_overrides else dict(ENV_OVERRIDES)
    scene_path = _resolve_g1_scene(repo_root)

    spec = mujoco.MjSpec.from_file(str(scene_path))
    torso = spec.body("torso_link")
    cam = torso.add_camera()
    cam.name = "head"
    cam.pos = HEAD_CAM_POS
    cam.quat = HEAD_CAM_QUAT

    world = spec.worldbody
    chase = world.add_camera()
    chase.name = "chase"
    chase.mode = mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM
    chase.targetbody = "torso_link"
    chase.pos = CHASE_CAM_POS
    chase.quat = look_at_quat(chase.pos, CHASE_CAM_TARGET)

    side = world.add_camera()
    side.name = "side"
    side.mode = mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM
    side.targetbody = "torso_link"
    side.pos = SIDE_CAM_POS
    side.quat = look_at_quat(side.pos, CHASE_CAM_TARGET)

    mj_model = spec.compile()
    mj_data = mujoco.MjData(mj_model)
    cam_ids = {
        "head": mj_model.camera("head").id,
        "chase": mj_model.camera("chase").id,
        "side": mj_model.camera("side").id,
    }
    renderer = mujoco.Renderer(mj_model, height=224, width=224)

    env = registry.load(env_name, config_overrides=env_overrides)

    policy = None
    if ckpt_policy is None:
        print(
            "[env] no policy_ckpt given — MPPI will plan without a PPO warm "
            "start (from zeros / receding-horizon)."
        )
    elif not ckpt_policy.exists():
        # A wrong path silently changes the experiment (no warm start). Paths
        # are resolved relative to the CWD, so a run launched from elsewhere is
        # the usual cause — surface it loudly rather than degrade silently.
        import warnings

        warnings.warn(
            f"policy_ckpt {ckpt_policy} does not exist — MPPI will plan WITHOUT "
            "a PPO warm start. Check the path (resolved relative to the CWD).",
            stacklevel=2,
        )
    else:
        if ckpt_policy.is_dir():
            params = brax_checkpoint.load(str(ckpt_policy.resolve()))
        else:
            with ckpt_policy.open("rb") as f:
                params = pickle.load(f)
        cfg = locomotion_params.brax_ppo_config(env_name)
        factory = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=tuple(
                cfg.network_factory.policy_hidden_layer_sizes
            ),
            value_hidden_layer_sizes=tuple(
                cfg.network_factory.value_hidden_layer_sizes
            ),
            policy_obs_key=cfg.network_factory.policy_obs_key,
            value_obs_key=cfg.network_factory.value_obs_key,
        )
        preprocess_fn = (
            running_statistics.normalize
            if cfg.normalize_observations
            else lambda obs, *_: obs
        )
        net = factory(
            env.observation_size,
            env.action_size,
            preprocess_observations_fn=preprocess_fn,
        )
        policy = jax.jit(
            ppo_networks.make_inference_fn(net)(params, deterministic=True)
        )

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    rng = jax.random.PRNGKey(seed)
    rng, rkey = jax.random.split(rng)
    state = reset_fn(rkey)

    return {
        "env": env,
        "state": state,
        "rng": rng,
        "reset_fn": reset_fn,
        "step_fn": step_fn,
        "mj_model": mj_model,
        "mj_data": mj_data,
        "cam_id": cam_ids["head"],  # back-compat
        "cam_ids": cam_ids,
        "renderer": renderer,
        "policy": policy,
    }


def _sync_mj_data(env_ctx: dict) -> None:
    """Copy mjx qpos/qvel to the CPU-side mj_data so we can render."""
    import mujoco

    state = env_ctx["state"]
    mjx_data = getattr(state, "data", None) or getattr(
        state, "pipeline_state", None
    )
    mj_data = env_ctx["mj_data"]
    mj_model = env_ctx["mj_model"]
    mj_data.qpos[:] = np.asarray(mjx_data.qpos)
    mj_data.qvel[:] = np.asarray(mjx_data.qvel)
    mujoco.mj_forward(mj_model, mj_data)


def render_head_cam(env_ctx: dict) -> np.ndarray:
    """Render the head-cam view at the current sim state. ``(H, W, 3)`` uint8."""
    _sync_mj_data(env_ctx)
    env_ctx["renderer"].update_scene(env_ctx["mj_data"], camera=env_ctx["cam_id"])
    return env_ctx["renderer"].render().copy()


def render_multi_cam(env_ctx: dict) -> np.ndarray:
    """Render head + chase + side cams horizontally stacked. Demo video only."""
    _sync_mj_data(env_ctx)
    renderer = env_ctx["renderer"]
    frames = []
    for name in ("head", "chase", "side"):
        renderer.update_scene(env_ctx["mj_data"], camera=env_ctx["cam_ids"][name])
        frames.append(renderer.render().copy())
    return np.concatenate(frames, axis=1)


def get_torso_xyz(env_ctx: dict) -> np.ndarray:
    """Read ``qpos[0:3]`` from the current state — torso COM."""
    state = env_ctx["state"]
    mjx_data = getattr(state, "data", None) or getattr(
        state, "pipeline_state", None
    )
    return np.asarray(mjx_data.qpos[:3])


def rollout_policy_warm_start(
    env_ctx: dict, horizon: int, rng_key
) -> tuple[np.ndarray | None, Any]:
    """Roll the loaded PPO policy on a SHADOW copy of the env state to
    produce a nominal action sequence of length ``horizon``.

    The real env state is never mutated. Returns
    ``(warm_U, next_rng_key)`` with ``warm_U (horizon, action_dim)``
    float32, or ``(None, rng_key)`` if no policy is loaded.
    """
    import jax

    policy = env_ctx.get("policy")
    if policy is None:
        return None, rng_key
    shadow_state = env_ctx["state"]
    pinned_cmd = env_ctx.get("pinned_cmd")
    acts: list[np.ndarray] = []
    for _ in range(horizon):
        rng_key, sub = jax.random.split(rng_key)
        action, _ = policy(shadow_state.obs, sub)
        acts.append(np.asarray(action, dtype=np.float32))
        shadow_state = env_ctx["step_fn"](shadow_state, action)
        if pinned_cmd is not None:
            shadow_state = shadow_state.replace(
                info={**shadow_state.info, "command": pinned_cmd}
            )
    return np.stack(acts, axis=0), rng_key


def load_goal_pixel(h5_path: Path, idx: int, ep_idx: int) -> np.ndarray:
    """Pick a mid-episode frame from the expert HDF5 as a goal image.

    Returns ``(224, 224, 3)`` uint8. The world-model target is
    visual-only, so the goal is a plain image (no proprio needed).
    """
    with h5py.File(h5_path, "r") as f:
        off = int(f["ep_offset"][ep_idx])
        L = int(f["ep_len"][ep_idx])
        t = min(off + L - 1, off + L // 2 + idx)
        return np.asarray(f["pixels"][t])


__all__ = [
    "ENV_NAME",
    "ENV_OVERRIDES",
    "build_env_and_policy",
    "get_torso_xyz",
    "install_jax_brax_shim",
    "load_goal_pixel",
    "render_head_cam",
    "render_multi_cam",
    "rollout_policy_warm_start",
]
