# Vitruvian — Roadmap

*Living document. Updated each session.*

**Last updated:** 2026-06-30

---

## North Star

Build a minimum viable humanoid and the machine-intelligence stack to inhabit
it, simulation-first, with self-learning as the defining research commitment.

The project takes a **graduated** approach: pretrained priors (visual
encoders, motion priors) are welcome where they sharpen the scientific
question — biological newborns are not tabula rasa. What matters is that the
*plastic* parts — world model, policy/planner, intrinsic drives, episodic
memory — genuinely self-organize from the agent's own experience.

See [`thesis.md`](thesis.md) for the long-form argument.

---

## 2026-06-30 — Reorientation (see [ADR 009](decisions/009-thesis-question-milestones.md))

The plumbing is done; the thesis is not. The roadmap pivots from
**engineering milestones** ("make it walk") to **thesis-question
milestones**, and the project **decouples the world model from task
reward**: intrinsic cost (homeostasis) + intrinsic motivation (curiosity /
model-disagreement / empowerment) + goal-conditioned planning — *not* a
reward-trained policy. Behavior comes from **planning through the world
model**. Metrics become **rollout accuracy** and **OOD-goal planning**, not
episode reward.

---

## Foundation — complete (the plumbing)

Condensed; full detail lives in the journal entries + ADRs 001–008, and in
git history prior to 2026-06-30.

| | |
|---|---|
| **Phase 0** | Paper trail — thesis, ADRs 001–005, journal. |
| **Phase 1** | G1 walks via PPO on Warp/MJX (M0–M1); reproducible; wandb preserved. |
| **Phase 2** | G1 hardening — rebranded per [ADR 006](decisions/006-g1-reference-body.md): Unitree G1 is the reference body. |
| **Phase 3** | DINOv3-backed JEPA world model + MPPI planner. M2 head-cam → M3 DINOv3 → M4.1–4.7 world-model arc → M4.8/4.9 installable library. (ADRs 007/008.) |
| **M5** *(2026-06-30)* | Review-driven correctness pass (rollout off-by-one, HER/IQL terminal, plan-time alignment, val-split leakage) + JEPA modernization: **SIGReg** anti-collapse (LeJEPA), **proprio-as-conditioning** with a visual-only target, DINOv3 kept over V-JEPA (FAIR planning study). Full reward-trained retrain on a cloud RTX PRO 6000 → this is the **baseline / control**. |

---

## Research phase — thesis-question milestones

Milestones are questions now, not features (see ADR 009).

### Now
- **Q1 — Does the planner earn its keep?**
  - (a) Open-loop **latent rollout accuracy** on held-out trajectories — the world model's true quality metric.
  - (b) MPPI success on **OOD goal-images** the PPO walker never optimized (sideways, backward, novel pose). If it only reproduces the expert, the world model is decorative.
  - *No new training — runs on the M5 baseline.*

### Soon
- **Q2 — Reward-free world model.** Zero `tracking_lin_vel`; keep an upright/homeostatic term; diversify (randomized commands + action noise) → data spanning *dynamics*, not the reward-optimal slice. Retrain JEPA; compare rollout accuracy + OOD planning to the baseline.
- **Q3 — Intrinsic motivation.** Drive exploration by the world model's own uncertainty (**Plan2Explore** / model-disagreement; or RND). MPPI-plan toward high-uncertainty states → the agent collects the data that most improves its own world model. This closed loop *is* the thesis. (The SIGReg latent gives a surprise signal to build on.)

### Later
- **Q4 — Hierarchy.** H-JEPA / HWM: multi-timescale world models for long-horizon latent planning (revisits [ADR 008](decisions/008-hwm-planning-layer.md); the shelved HWM is now a published method).
- **Q5 — Continual learning.** Retire the offline collect→train→eval batch loop for a live **act → observe → update** loop — the lifetime written by the agent's own experience. Then episodic memory, then language.

---

## Cadence

Each session: start by reading the latest journal entry + this file; end with a
commit, a new journal entry, and a roadmap update. Decisions worth a
six-month-later explanation get a new ADR.

---

## Locked decisions

1. [ADR 001 — `uv` env manager](decisions/001-env-manager-uv.md)
2. [ADR 002 — Monorepo + vendored submodules](decisions/002-repo-structure-monorepo.md)
3. [ADR 003 — Graduated Vitruvian architecture](decisions/003-graduated-vitruvian.md)
4. [ADR 004 — Weights & Biases](decisions/004-experiment-tracking-wandb.md)
5. [ADR 005 — Apache 2.0](decisions/005-license-apache-2.md)
6. [ADR 006 — Unitree G1 reference body](decisions/006-g1-reference-body.md)
7. [ADR 007 — LeWM plastic world model](decisions/007-lewm-world-model.md)
8. [ADR 008 — HWM planning layer](decisions/008-hwm-planning-layer.md)
9. [ADR 009 — Thesis-question milestones, reward-decoupled](decisions/009-thesis-question-milestones.md)

---

## Hardware

| | |
|---|---|
| **Local** | RTX 4060 Ti 8 GB (Ada), i7-14700F, 31 GiB. The 8 GB is the dev constraint. |
| **Cloud** | Lightning.ai — interruptible **RTX PRO 6000 Blackwell** (96 GB, 500 TFLOPs, 48 CPU) for >1 h jobs; H100 available. Small tasks local; big tasks cloud. |

---

## Open questions carried forward

- Which intrinsic signal first — RND vs Plan2Explore vs empowerment?
- Does the SIGReg latent's surprise double as the exploration drive?
- At what planning-horizon threshold does hierarchy become necessary?
- When (if ever) does a "frozen" prior become unfrozen, and on what signal?
- When does language enter the stack, and how?
- Cross-embodiment transfer (G1 → ToddlerBot / H1 / Apollo)?
