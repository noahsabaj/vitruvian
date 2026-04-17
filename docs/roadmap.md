# Vitruvian — Roadmap

*Living document. Updated each session.*

**Last updated:** 2026-04-16

---

## North Star

Build a minimum viable humanoid and the machine-intelligence stack to inhabit
it, simulation-first, with self-learning as the defining research commitment.

The project takes a **graduated** approach: pretrained priors (visual
encoders, motion priors) are welcome where they sharpen the scientific
question, on the understanding that biological newborns are not tabula
rasa — they carry hundreds of millions of years of evolutionary firmware.
What matters is that the *plastic* parts of the stack — the world model,
the policy, the intrinsic drives, the episodic memory — genuinely
self-organize from the agent's own experience.

See [`thesis.md`](thesis.md) for the long-form argument.

---

## Phases

### Phase 0 — Paper Trail *(current)*

Write down the plan, the thesis, and the architectural decisions before
writing any code. The paper trail *is* the plan.

**Artifacts:**
- `README.md`, `LICENSE`, `.gitignore`
- `docs/roadmap.md` (this file)
- `docs/thesis.md`
- `docs/decisions/001` through `005`
- `docs/journal/2026-04-16.md`

**Exit criteria:** All of the above exist and are committed to an initial
local git repo.

---

### Phase 1 — Simulation Infrastructure *(complete, 2026-04-16)*

Prove the entire simulation + RL pipeline end-to-end on a known-good
humanoid, on local hardware, reproducibly. No Vitruvian-specific modeling
yet.

GPU backend: **NVIDIA Warp** (via `mujoco_warp`) is the default in
`mujoco_playground` 0.2.0 (March 2026) and roughly 250× faster than
MJX-JAX on locomotion. We install both — Warp for perf, MJX for
gradient support and non-NVIDIA portability.

| Milestone | Status | Description |
|---|---|---|
| **M0.1** | ✓ | `uv` project initialized, Python 3.12, monorepo layout, docs skeleton, first commit, pushed to private GitHub. |
| **M0.2** | ✓ | `mujoco`, `mujoco-mjx`, `mujoco-warp`, `warp-lang`, `jax[cuda12]`, `brax` installed. JAX and Warp both see the RTX 4060 Ti. `mujoco_menagerie` added as submodule. |
| **M0.3** | ✓ | Unitree G1 loads in the MuJoCo viewer, random torques applied, physics confirmed, screen capture saved. |
| **M0.4** | ✓ | `mujoco_playground` installed, G1 locomotion environment instantiates with the Warp backend, random policy rolls out successfully. |
| **M1**   | ✓ | PPO trained G1 to walk (eval_reward +3.70 at 43 M steps, user-confirmed walking with no falls). Training reproducible via a single command. wandb runs preserved ([smoke](https://wandb.ai/noahsabaj-myself/vitruvian/runs/2be0vxgu), [full](https://wandb.ai/noahsabaj-myself/vitruvian/runs/oo2h5y6t)). Video of the trained policy saved to `docs/journal/assets/2026-04-16-m1-g1-full.gif`. |

**Phase 1 exit criteria met** — video committed, wandb dashboards
bookmarked, full story in [`docs/journal/2026-04-16.md`](journal/2026-04-16.md).

---

### Phase 2 — G1 Hardening *(rebranded after [ADR 006](decisions/006-g1-reference-body.md))*

Originally: "Design the MVH (20 DoF, 60-70 cm, hobby-servo class)" —
**superseded by ADR 006**. Unitree G1 is Vitruvian's reference body
for the foreseeable future; we are not designing our own robot.

In its place, a lighter pre-Phase-3 set of optional milestones on G1.
None of these are blockers for Phase 3 — Phase 3 can begin immediately
on the current stack.

- **Close the unfinished M1 loop.** Rerun the 100 M-step training on
  G1 with VRAM-hygiene fixes (drop `num_envs` to 768 or render viz
  out-of-process) so training completes cleanly without the
  step-48.66 M OOM crash we hit in M1-full. Deliverable: a
  fully-converged PPO walker at 100 M steps.
- **Rough-terrain robustness.** Train on `G1JoystickRoughTerrain` —
  same script, different env — to verify the PPO pipeline generalizes
  past flat floor.
- **FastTD3 baseline.** Per [ADR 003 addendum](decisions/003-graduated-vitruvian.md):
  FastTD3 is the 2026 SOTA humanoid-locomotion algorithm and beats
  PPO/SAC/TD-MPC2/DreamerV3 on wall-clock. Worth carrying as a
  parallel training track before or during Phase 3.

Milestones picked on demand, not committed in advance.

---

### Phase 3 — World Models on G1 *(current focus after Phase 1 closure)*

Replace the policy-only baseline with **Dreamer 4** (arXiv 2509.24527,
Sep 2025 — the current Hafner-lineage SOTA, superseding Dreamer V3),
with **TD-MPC2** carried as a parallel track. Mount a simulated head
camera on G1; attach **DINOv3** (dense spatial, commercial license)
and **V-JEPA 2.1** (temporal, arXiv 2603.14482, Mar 2026,
CC-BY-NC-ND — `+20pt` real-robot grasping vs V-JEPA 2 AC) as
complementary frozen visual encoders. This is where the graduated
architecture lands in earnest: frozen evolutionary-equivalent priors +
plastic world model running continuous self-supervised prediction.

See [ADR 003](decisions/003-graduated-vitruvian.md) (with the
2026-04-16 addendum) for candidate lists.

**Staging (one milestone per session, roughly):**

| Milestone | What |
|---|---|
| **M2** | Add a simulated head camera to the G1 env; render its feed during training; confirm observations flow through the pipeline. |
| **M3** | Wire up **DINOv3** (dense / spatial) and/or **V-JEPA 2.1** (temporal, +20pt real-robot grasping vs V-JEPA 2) as a frozen visual encoder; confirm we can process camera observations through it and surface the latent to the policy. |
| **M4** | Replace PPO with **Dreamer 4** (or TD-MPC2); retrain G1 using the world-model loop with the visual latent as part of the observation. |
| **M5** | Reward ablation: how much of G1's walking behavior survives with the `tracking_lin_vel` reward zeroed out and only self-supervised prediction + intrinsic-motivation signals driving exploration? First test of the self-learning commitment. (Bridges into Phase 4.) |

---

### Phase 4 — Self-Learning

Intrinsic motivation (RND / Plan2Explore / empowerment), episodic memory,
long-horizon and sparse-reward tasks. The research question proper begins
here.

Not scoped in detail yet.

---

### Phase 5 — Hardware *(deferred indefinitely after [ADR 006](decisions/006-g1-reference-body.md))*

Per ADR 006, Vitruvian does not build its own robot for the
foreseeable future. Unitree G1 is our reference body. Hardware is
revisited only if a credible universal open-humanoid design emerges,
or if the user's ability to contribute mechanical design changes.

If/when Phase 5 is reopened, the original plan was:

- 3D-printed structure, Dynamixel XL330-class servos, RPi 5 onboard,
  sim-to-real via domain randomization.
- **Primary design reference:** ToddlerBot (Stanford, CoRL 2025).
  Closest public match to the original Vitruvian target envelope
  (0.56 m, 3.4 kg, 30 DoF, fully 3D-printed, <$6k). Source:
  `hshi74/toddlerbot`. Present in sim as
  `external/mujoco_menagerie/toddlerbot_{2xc,2xm}`.
- **Secondary:** Berkeley Humanoid Lite (RSS 2025,
  `HybridRobotics/Berkeley-Humanoid-Lite`). BLDC-class, larger
  (~0.8 m, 16 kg), BOM ~$4.3 k. Useful for cycloidal-gearbox patterns
  if/when we graduate from hobby servos.
- **Dead end:** K-Scale Labs / Zeroth Bot shut down Nov 2025. Repos
  still at `kscalelabs` under CERN-OHL-S-2.0 / MIT for archaeological
  reference.

---

## Cadence

Regular work sessions (evenings / weekends). Each session:

1. Starts by reading the most recent journal entry and the current roadmap
   state.
2. Ends with: a commit, a new journal entry, and a roadmap update if
   anything moved.

Architectural decisions that would deserve a six-month-later explanation
get written as a new ADR in `docs/decisions/`.

---

## Local hardware context

| | |
|---|---|
| **GPU** | NVIDIA RTX 4060 Ti, 8 GB VRAM (Ada Lovelace) |
| **CPU** | Intel i7-14700F, 20 cores / 28 threads |
| **RAM** | 31 GiB |
| **Disk** | 691 GB (~177 GB free at project start) |
| **OS** | Linux Mint 22.3 (Ubuntu 24.04 noble base) |
| **CUDA** | Toolkit 12.8, driver supports 13.0 |

The 8 GB VRAM is the first real constraint to watch in Phase 3+.

---

## Locked decisions

1. [ADR 001 — `uv` as environment manager](decisions/001-env-manager-uv.md)
2. [ADR 002 — Monorepo with vendored submodules](decisions/002-repo-structure-monorepo.md)
3. [ADR 003 — Graduated Vitruvian architecture](decisions/003-graduated-vitruvian.md)
4. [ADR 004 — Weights & Biases for experiment tracking](decisions/004-experiment-tracking-wandb.md)
5. [ADR 005 — Apache 2.0 license](decisions/005-license-apache-2.md)
6. [ADR 006 — Unitree G1 as Vitruvian's reference body](decisions/006-g1-reference-body.md)

---

## Open questions carried forward

- At what point does language enter the stack, and how?
- Which self-learning signal to prioritize in Phase 4: RND vs
  Plan2Explore vs empowerment vs LLM-driven intrinsic reward
  (VSIMR+LLM, IMAGINE, MERCI)?
- When (if ever) does a "frozen" prior become unfrozen, and on what signal?
- Current plan: **DINOv3** (dense spatial, commercial license) +
  **V-JEPA 2.1** (temporal, CC-BY-NC-ND, +20pt real-robot grasping
  vs V-JEPA 2 AC per arXiv 2603.14482, Mar 2026). Open: does VL-JEPA
  or some later JEPA-world model eventually replace the pair?
- Dreamer 4 reimplementation choice: `nicklashansen/dreamer4` (PyTorch,
  DMControl-targeted) vs `lucidrains/dreamer4` vs wait for an official
  Hafner release.
- **Cross-embodiment transfer** (new, from [ADR 006](decisions/006-g1-reference-body.md)):
  methods developed on G1 should in principle transfer to ToddlerBot,
  H1, Apollo, etc. When does that enter the agenda explicitly? Probably
  Phase 4 when policies are expected to generalize beyond training
  distribution.
- If/when a universal open humanoid design is published under a
  permissive license, does Phase 5 reopen?
