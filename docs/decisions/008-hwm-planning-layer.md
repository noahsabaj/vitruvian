# ADR 008 — Hierarchical World Models (HWM) as the Planning Layer

**Status:** Accepted
**Date:** 2026-04-17

## Context

[ADR 007](007-lewm-world-model.md) adopted **LeWM** as Vitruvian's
plastic world model. M4.1 → M4.3 proved:

- LeWM installs and runs on our 4060 Ti (M4.1)
- Our M1 PPO walker yields a clean 20k-step G1 expert dataset (M4.2)
- LeWM's JEPA learns G1 dynamics from that dataset — `validate/pred_loss`
  dropped 0.074 → 0.031 (-58 %) in 30 training batches on the smoke
  (M4.3)

The pending step was M4.4: actually *plan* with the trained world
model to produce a walking policy. I flagged this during the session
as "genuinely hard research, 2-4 week arc," with three specific
reasons:

1. Single-level CEM accumulates prediction error on long horizons.
   G1 walking is a long-horizon task (500+ control steps per minute of
   walk); a pixel-level JEPA predictor will drift.
2. The search space over a 29-dimensional continuous action at 50 Hz
   is exponentially bigger than PushT's 2-D pushes.
3. LeWM's default paradigm is goal-image-based cost. "Track a
   joystick velocity command" is not a goal-image problem.

On 2026-04-17, mid-session, the user surfaced a paper submitted only
two weeks prior (3 April 2026):

> **Hierarchical Planning with Latent World Models**
> Wancong Zhang, Basile Terver, Artem Zholus, Soham Chitnis, Harsh Sutaria, Mido Assran, Randall Balestriero, Amir Bar, Adrien Bardes, Yann LeCun, Nicolas Ballas.
> arXiv:2604.03208. CC BY 4.0.
> Code: https://github.com/kevinghst/HWM_PLDM

The abstract motivation is a verbatim match for the blockers above:

> "Model predictive control (MPC) with learned world models has
> emerged as a promising paradigm for embodied control… However,
> learned world models often struggle with long-horizon control due to
> the accumulation of prediction errors and the exponentially growing
> search space. In this work, we address these challenges by learning
> latent world models at multiple temporal scales and performing
> hierarchical planning across these scales…"

Headline results:

| Task | Baseline (single-level) | HWM (hierarchical) | Δ |
|---|---:|---:|---|
| Franka pick-and-place (no subgoals) | 0 % (V-JEPA 2-AC) | **70 %** | +70 |
| Franka drawer | 30 % | 70 % | +40 |
| Push-T, horizon 75 | 17 % (DINO-WM) | 61 % | +44 |
| Diverse Maze hard | 44 % (PLDM) | 83 % | +39 |

And 3-4× less planning-time compute on the physics-based simulated
tasks. Outperforms VLAs (Octo, π₀-FAST-DROID, π₀.₅-DROID) trained on
~77× more robotic data.

Most importantly for us: HWM is **architecture-agnostic**. The paper
demonstrates it as a plug-in planning layer on top of three different
latent-world-model families (V-JEPA 2-AC, DINO-WM, PLDM) without
modifying any of them. LeWM is architecturally close to DINO-WM /
PLDM (a simple predictor on top of a JEPA-style encoder); plugging
LeWM in is the natural extension.

## Decision

**Vitruvian adopts HWM as the planning layer for M4.4 and beyond.**
The trained LeWM from M4.3 becomes the **low-level world model** in
the hierarchy. A separately-trained **high-level macro-action world
model** will be added on top, sharing LeWM's frozen encoder. Both
levels use Cross-Entropy Method (CEM) for action optimization; the
high-level planner outputs subgoals in the encoder's latent space,
the low-level planner chases them with primitive 29-D actions at 50
Hz.

Concretely:

- **Low-level world model:** LeWM (JEPA encoder + single-step
  predictor over primitive actions). Already trained — M4.3
  checkpoint at `~/.stable_worldmodel/lewm_g1_v1_weights.ckpt`.
- **High-level world model:** a macro-action world model trained on
  waypoint-segmented G1 expert trajectories. Encoder is frozen
  (LeWM's ViT-tiny), so training is just the high-level predictor
  plus an action encoder mapping primitive-action subsequences to
  macro-actions. Per the HWM paper, macro-action latent dim ≈ 4.
- **Goal specification:** a **goal image** of G1 at the target
  spatial configuration (e.g., the expert walker 10 m forward from
  the current pose). LeWM's encoder embeds the goal; CEM cost is L1
  distance in that latent space.
- **Planner:** top-down hierarchical CEM following HWM §3.
  - High level: optimize macro-action sequence to reach
    `z_goal = E(goal_image)`.
  - The first predicted subgoal latent becomes the low-level target.
  - Low level: optimize primitive actions (29-D, N-step horizon) to
    reach the subgoal latent.
- **Codebase reference:** `kevinghst/HWM_PLDM` vendored as a git
  submodule under `external/hwm` when we start M4.4 proper. We expect
  to reuse their CEM + hierarchy scaffolding, swap LeWM for PLDM as
  the low-level.

## Consequences

- **M4.4 becomes a concrete staged arc, not blue-sky research.** Sub-
  milestones:
  - **M4.4a** — vendor HWM, run their PLDM+maze smoke on our box to
    verify install.
  - **M4.4b** — segment our 20k-step G1 dataset into macro-action
    waypoint trajectories; train the high-level macro-action WM on
    top of LeWM's frozen encoder (~1-3 hours wall on our 4060 Ti).
  - **M4.4c** — implement hierarchical CEM with LeWM + macro-action
    WM. Smoke-test by planning toward a goal image from one of our
    expert rollouts (should reconstruct forward walk).
  - **M4.4d** — full eval: novel goal images (targets the policy
    didn't see during dataset collection), success criterion = G1
    walks toward goal without falling, measured via playground
    physics.
- **LeWM training already done is not wasted.** The M4.3 checkpoint
  *is* the low-level WM of the hierarchy. We don't restart.
- **Research framing sharpens.** The novel contribution becomes:
  - First known application of HWM to **humanoid locomotion**
    (paper's demonstrated tasks are manipulation and planar maze,
    not 29-DoF contact-rich balance).
  - First known application of HWM on top of a **LeWM-style
    JEPA-only world model** (paper uses DINO-WM + PLDM; LeWM is a
    newer JEPA variant).
- **Goal-image design matters.** Our M4.2 dataset contains expert
  rollouts at various joystick commands. Goal images for eval can be
  sourced from: (a) held-out rollouts from the PPO walker, or (b)
  synthesized by rolling the PPO walker to target positions we care
  about. Pick when we get to M4.4d.
- **Compute budget is manageable.** HWM high-level training on top of
  a frozen encoder is much cheaper than the low-level LeWM run we
  just did (the ViT is frozen; only the small macro-action
  predictor is trained). ~1-3 hours for a full high-level epoch.
- **License composition.** HWM is CC BY 4.0. Our Apache-2.0 main repo
  is compatible; we note HWM code as a CC-BY dependency per ADR 005's
  compatibility discipline.
- **Open research risk.** HWM has no humanoid / locomotion results.
  Contact-rich 29-DoF dynamics may surface failure modes the
  paper's pick-&-place and maze experiments can't predict. Accept
  that risk — the downside is "we learn what breaks," not "we waste
  work," because HWM vs single-level is a drop-in replacement.

## Alternatives Considered

- **Stick with single-level LeWM + vanilla CEM.** Rejected: the
  paper's 0 % vs 70 % Franka result is strong evidence this is the
  wrong regime for any long-horizon task. Our humanoid is worse
  than pick-&-place along every relevant dimension.
- **Swap LeWM for V-JEPA 2-AC + HWM** (as the paper does for their
  Franka experiments). Rejected for now: V-JEPA 2-AC is a 1.2 B-param
  video model, tuned for RTX 4090 at minimum. Doesn't fit our 8 GB
  card. Revisit if we rent cloud compute.
- **Defer HWM adoption to Phase 4** and run a naive CEM first.
  Rejected: naive CEM was already flagged as multi-session research;
  HWM is also multi-session but produces a result with strong
  published precedent. Spending the same calendar time on the
  paved path is straightforwardly better.

## Open Questions

- How are macro-actions defined for humanoid locomotion? The paper's
  high-level action encoder maps subsequences of primitive actions
  into a learned continuous macro-action. For G1 at 50 Hz with
  waypoints every ~1 s, each macro-action spans ~50 primitive
  actions. Latent dim ~4 per the paper; will that bottleneck our
  walking diversity? Expect to need tuning.
- How many waypoints per episode? Pick-&-place has ~2-3 natural
  waypoints (grasp, move, place). G1 joystick tracking is more
  uniform — maybe one waypoint per second of walking. TBD during
  M4.4b.
- Does the goal-image + L1-in-latent cost produce a smooth
  optimization surface for walking? Pick-&-place has a well-defined
  terminal state; "walking forward 10 m" is more of a trajectory
  than a point. We may need to swap to a directional cost (velocity
  latent match) if pure goal-matching plateaus.
- Cross-embodiment transfer is still deferred (from ADR 006's open
  questions). HWM's architecture-agnosticism suggests the method
  transfers; the specific high-level model won't.
