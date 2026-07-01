# ADR 009 — Reorient Around Thesis-Question Milestones (World-Model-First, Reward-Decoupled)

**Status:** Accepted
**Date:** 2026-06-30

## Context

Phases 0–3 (M0–M4.9) built the plumbing: G1 sim on Warp/MJX, a PPO
walker, a frozen DINOv3 prior, a JEPA world model, an MPPI planner, and
an installable library. This session's **M5** pass hardened it — a
review-driven correctness sweep (rollout-target off-by-one, HER/IQL
terminal handling, plan-time history/action alignment, train/val
leakage) plus a research-informed modernization (SIGReg anti-collapse
per LeJEPA; proprio-as-*conditioning* with a visual-only prediction
target; DINOv3 retained over V-JEPA per FAIR's planning study
arXiv:2512.24497). The plumbing works.

The **thesis** does not yet stand. Its core commitment — *a plastic
world model that self-organizes from the agent's own experience, with
self-learning as non-negotiable* — is unproven, for two structural
reasons surfaced in a critique framed around Yann LeCun's public
positions (fitting, since the stack is built almost entirely on his
lab's output: JEPA, LeJEPA/SIGReg, DINOv3, LeWM):

1. **We lean on task reward.** The world model trains on rollouts from a
   `tracking_lin_vel`-optimized PPO walker, so it only knows the narrow
   reward-optimal manifold — not the general dynamics needed to plan
   novel behavior.
2. **The planner may not earn its keep.** We have not shown JEPA+MPPI
   does anything the policy that generated its data cannot.

The `M1…M9` framing is engineering-deliverable-shaped ("make it walk").
Right for bootstrapping; wrong for the research phase.

## Decision

Reorient the roadmap from **engineering milestones** to
**thesis-question milestones**, and **decouple the world model from task
reward** — replace the external task reward (`tracking_lin_vel`) with
*intrinsic cost* (homeostasis: stay upright / don't self-damage) +
*intrinsic motivation* (curiosity / model-disagreement / empowerment) +
*goal-conditioned planning*. Behavior is produced by planning through
the world model, not by a reward-trained policy.

This is **not "no objective"** — it is "no task-specific external
reward." Fully-from-zero humanoid control is impractical; a minimal
homeostatic prior + intrinsic drives is the workable form of
"reward-free."

Organizing questions (metrics are **rollout accuracy** and **OOD-goal
planning**, not episode reward):

- **Q1 (now):** Does the planner beat the policy that trained it, on
  *out-of-distribution* goals? + open-loop latent rollout accuracy as
  the world model's true metric. *No new training — runs on the M5
  baseline.*
- **Q2 (soon):** Can the world model be learned **reward-free** (task
  reward zeroed, upright-only + diversified data) and still plan as
  well / better?
- **Q3 (soon):** Does **intrinsic motivation** (Plan2Explore-style
  model-disagreement, or RND) drive self-improving data collection
  faster than random?
- **Q4 (later):** Does **hierarchy** (H-JEPA / HWM, multi-timescale
  latent planning) extend the planning horizon?
- **Q5 (later):** Can it learn **continually / online** from its own
  experience — retiring the offline collect→train→eval batch loop?

## Consequences

- The reward-trained M5 run becomes the **baseline / control**, not the
  destination.
- New metrics: open-loop rollout accuracy + OOD-goal planning success,
  not episode reward.
- Milestones are now scientific questions; the ADR + journal discipline
  is retained (the one thing the critique explicitly endorsed keeping).
- **Real risk:** reward-free world-model learning on a 29-DoF
  contact-rich body may not yield competent planning (ADR 007's risk,
  now central). A negative result — "JEPA breaks *here* on contact-rich
  locomotion" — is itself a documented, valuable finding.

## Alternatives Considered

- **Keep engineering milestones, optimize walking.** Rejected: plumbing
  is done; a better reward-walker doesn't test the thesis.
- **Full model-free online RL.** Rejected: the thesis is world-model +
  planning, not model-free RL.
- **Fully-from-zero (no objective at all).** Rejected as impractical for
  a humanoid; minimal homeostasis + intrinsic drives is the pragmatic
  "reward-free."

## Open Questions

- Which intrinsic signal first — model-disagreement (Plan2Explore) vs
  RND vs empowerment?
- Does the SIGReg latent's surprise double as the exploration signal?
- At what planning-horizon threshold does hierarchy become necessary?
- When does language enter (unchanged open question)?
