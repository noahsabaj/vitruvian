# ADR 007 — LeWorldModel (LeWM) as Vitruvian's Plastic World Model

**Status:** Superseded (amended 2026-04-23, M4.9)
**Date:** 2026-04-17

> **Amendment (M4.9, 2026-04-23):** The `external/le-wm` submodule is
> retired. The ~550 LOC of LeWM classes we actually depend on
> (`Embedder`, `ARPredictor`, `MLP`, `Transformer` blocks, `SIGReg`,
> and the legacy composer) are vendored in
> `src/vitruvian/lewm_compat/` under Apache-2.0 attribution (see
> `NOTICE` and `docs/archive/lewm_local_edits.patch`). The unified
> `vitruvian.models.JEPA` composer + `build_jepa`/`load_jepa`
> registry subsumes v3/v4/v5 JEPA shapes; LeWM's scratch-trained
> ViT-tiny v3 path is kept loadable (via
> `load_lewm_jepa_from_checkpoint`) for historical ckpts, but the
> active world model since M4.5 has been DINOv3-backed (see
> ADR 003 addendum). The original decision below is preserved for
> its rationale; the implementation moved on.

## Context

[ADR 003](003-graduated-vitruvian.md) committed Vitruvian to the "graduated
architecture": frozen evolutionary-equivalent priors + plastic world
model + self-learning. The 2026-04-16 addendum named **Dreamer 4** as
the primary Phase-3 world-model candidate and **TD-MPC2** as the
parallel track.

Phase 1 closed on G1 walking via PPO. Phase 3 (M2 + M3) wired a
head-mounted camera through DINOv3 and produced per-frame frozen
latents. The next step was to replace PPO with a world-model
algorithm consuming those latents. When we went to execute, two hard
facts emerged:

1. **Dreamer 4 is hardware-incompatible.** The canonical unofficial
   port (`nicklashansen/dreamer4`) requires 8 GPUs × ≥24 GB VRAM each
   and >256 GB RAM. We have one 4060 Ti × 8 GB and 31 GB RAM. It's
   not tight; it's infeasible by two orders of magnitude.
2. **TD-MPC2 is from October 2023**, 2.5 years old, and the user
   explicitly objected to a 3-year-old method. Its descendants
   (Newt, TD-M(PC)², Puppeteer) are closer but share its MPPI/CEM
   test-time-planning DNA.

A fresh research sweep on 2026-04-17 surveyed the Oct 2025 — April
2026 window for world-model methods compatible with single-GPU
single-task continuous control. The candidate set:

| Method | Released | Humanoid-tested | Fits 8 GB | Paradigm |
|---|---|---|---|---|
| LeWM / LeWorldModel | Mar 2026 | No (PushT, reacher, cube, two-room) | Yes (15 M params) | JEPA world model + CEM planning, offline + test-time |
| Newt | Nov 2025 | Yes (DMControl humanoid) | Yes | TD-MPC2 descendant, language-conditioned multitask |
| WIMLE | Feb 2026 | Yes (8/14 HumanoidBench) | Likely | Uncertainty-aware world model |
| TD-M(PC)² | Feb 2025 | Yes (61-DoF humanoid) | Yes | TD-MPC2 + policy-constraint |
| Puppeteer | May 2024 (ICLR 2025) | Yes (CMU humanoid) | No (requires ≥24 GB) | TD-MPC2 hierarchical |
| BeyondMimic | Aug 2025 | Real Unitree G1 | Likely | Diffusion motion prior + RL |

## Decision

**Vitruvian adopts LeWM as the primary plastic-world-model component
for Phase 3 and beyond.**

- Repo: [`lucas-maes/le-wm`](https://github.com/lucas-maes/le-wm),
  arXiv 2603.19312, March 2026.
- Vendored under `external/le-wm` as a git submodule with its own
  isolated Python 3.10 venv (avoids conflict with our main 3.12
  environment).
- Datasets fetched from
  `https://huggingface.co/collections/quentinll/lewm`.

## Why LeWM specifically

The project's thesis — "plastic world model + self-supervised
prediction on top of frozen evolutionary-equivalent priors" — is
exactly what JEPA is. LeWM is the first JEPA-family world model
demonstrated to train stably end-to-end on pixel inputs with a
competitive planner. Picking LeWM makes the graduated-Vitruvian
architecture concretely realizable rather than aspirational.

Secondary reasons:

- **Single-GPU friendly.** 15 M params, `tiny` encoder by default, bf16
  training; validated paper claim of "one GPU, few hours." Fits our
  4060 Ti (after batch-size tuning — see M4 journal entry for the
  OOM-at-128 → works-at-32 path).
- **One month old (March 2026).** Strictly newer than the user's
  three-year cutoff; satisfies the "I want something current" bar.
- **LeCun / FAIR lineage.** Same direction as V-JEPA 2 / 2.1 / VL-JEPA,
  all of which we already acknowledged as our frozen-encoder track.
- **Permissive license** (MIT per the repo LICENSE file).

## Consequences

- **Phase 3 becomes a multi-session research arc,** not a single-session
  engineering task. LeWM has never been applied to humanoid
  locomotion; we are extending it. Realistic horizon: 2-4 weeks of
  focused work across several sessions.
- **Paradigm shift from online RL to offline + planning.** LeWM trains
  its world model on a pre-collected expert dataset, then uses CEM /
  MPC to plan against the learned dynamics at eval time. For G1
  adaptation we will need to:
  1. Collect a G1 expert dataset by rolling out the M1 PPO walker
     (the checkpoint at `checkpoints/m1-g1-full/000043253760`, eval
     reward +3.70).
  2. Pack trajectories into the HDF5 format LeWM expects
     (`pixels, action, proprio, state` keys, per
     `stable_worldmodel.data.HDF5Dataset`).
  3. Register a G1 wrapper under `stable_worldmodel.envs`.
  4. Write Hydra train + eval configs for G1 joystick-tracking.
- **Risk acknowledged.** LeWM is one month old research code validated
  on 2-D planar and simple manipulation tasks. 29-DoF contact-rich
  bipedal locomotion is a qualitatively different regime. Multiple
  failure modes exist; none are resolved by engineering alone.
  - World-model may fail to learn stable dynamics on contact-rich
    scenes.
  - CEM / MPC planner may not scale to a 29-D continuous action
    space at 50 Hz control rate.
  - Joystick velocity conditioning is not a native LeWM feature.
- **DINOv3 relationship remains open.** LeWM trains its own ViT-tiny
  encoder end-to-end on pixels. Three paths to reconcile with
  ADR 003's "frozen DINOv3" commitment:
  - (a) Accept LeWM stock for the first walking attempt; honor
    ADR 003 in a follow-on experiment. *Currently selected.*
  - (b) Modify LeWM to consume DINOv3 features in place of pixels.
  - (c) Hybrid: both signals into the planner / policy.
- **Deferred work.** Newt and WIMLE are carried as fallback options if
  LeWM does not extend cleanly to G1. Puppeteer stays on the
  Phase-5-hardware-reference track only (24 GB VRAM requirement
  excludes it from current work).

## Alternatives Considered

- **Dreamer 4** (arXiv 2509.24527) — rejected: hardware-infeasible.
  Revisit only if we rent cloud compute.
- **TD-MPC2** (Oct 2023) — rejected: user objected to age; descendants
  below are strictly newer.
- **Newt** (Nov 2025) — rejected as *primary* but carried as
  fallback. Strong humanoid validation but TD-MPC2-lineage rather
  than JEPA-lineage; doesn't advance the thesis.
- **WIMLE** (Feb 2026) — kept as fallback. Good HumanoidBench numbers
  but not JEPA-family.
- **Puppeteer** (ICLR 2025) — rejected: 24 GB VRAM requirement.
- **BeyondMimic** (Aug 2025) — interesting (real Unitree G1 + diffusion
  motion prior) but different paradigm (not a world model in the
  JEPA sense). Tracked as a reference for Phase 4 / 5.

## Open Questions

- Will the LeWM training loop produce a coherent world model on G1
  rollouts from our M1 PPO walker? First training run will answer.
- Will CEM planning in the learned latent recover walking behavior?
- Does the learned world model generalize enough for joystick
  commands outside the expert-distribution trained on?
- How many expert-trajectory episodes do we need? LeWM on PushT
  trained on ~50k steps of expert data — we'll need a comparable or
  larger G1 rollout corpus.
