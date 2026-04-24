# Vitruvian — Roadmap

*Living document. Updated each session.*

**Last updated:** 2026-04-23

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

### Phase 3 — World Models on G1 *(complete, 2026-04-23)*

Replaced the policy-only baseline with a DINOv3-backed JEPA world
model driving an MPPI planner. Originally scoped as Dreamer 4 +
TD-MPC2; landed on LeWM (M4.1–M4.3) → Chinchilla-scaled JEPAv4
(M4.5) → patch-latent JEPAv5 (M4.6) → unified library (M4.8/M4.9).
This is where the graduated architecture arrived: frozen
evolutionary-equivalent prior (DINOv3) + plastic world-model stack
(proprio encoder, action encoder, patch predictor) + MPPI planner
on top.

See [ADR 003](decisions/003-graduated-vitruvian.md) for candidate
lists and [ADRs 007 / 008](decisions/) for the world-model +
planning-layer decisions.

**Completed milestone history:**

| Milestone | Status | What shipped |
|---|---|---|
| **M2** | ✓ 2026-04-17 | Simulated head camera on G1 torso; rendered view flows through training pipeline. |
| **M3** | ✓ 2026-04-17 | DINOv3 ViT-B/16 (Meta, Apache-2.0) as the frozen visual encoder; ImageNet-normalized 224² pipeline; CLS + patch paths both tested. |
| **M4.1–M4.3** | ✓ | LeWM low-level world model on 20k-step G1 expert dataset. Plumbing proven; no steering signal yet (CLS pose-invariance issue). |
| **M4.4** | ✓ | HWM hierarchical planning + flat MPPI baseline on primitives. HL head didn't improve planning; flat MPPI proved the plumbing end-to-end. |
| **M4.5** | ✓ | Chinchilla-scaled rebuild: DINOv3 ViT-B/16 backbone, 38M-param predictor, 270k-transition diverse dataset, 6-step rollout supervision. `val_pred` fell 13.5× vs M4.4. |
| **M4.6** | ✓ | Patch-latent JEPAv5 (7×7 subsampled) to recover pose discrimination that CLS couldn't provide. First measurable MPPI steering signal. |
| **M4.7** | ✓ | Unified infrastructure refactor: EmbeddingCache (mmap), compile_utils (BF16 + torch.compile), EncoderHistory (encode-once), shared-AdaLN predictor, single-process eval driver. End-to-end cycle ≤ 50% of pre-refactor wall time. |
| **M4.8** | ✓ 2026-04-23 | Installable library: unified `JEPA` + registry, `MPPIPlanner` + cost strategies, `JEPATrainer`, `vit-*` CLI + YAML configs, 36 CPU tests, LeWM vendored. M4.4 HWM research code + 9 legacy scripts deleted; hwm/ shim layer removed. |
| **M4.9** | ✓ 2026-04-23 | Finish-the-polish: inline DINOv3 precompute into vit-train; port collect orchestrator; write `scripts/migrate_ckpt.py`; retire `external/le-wm` + `external/hwm` submodules. |
| **M4.9.1** | ✓ 2026-04-23 | Cold-review cleanup pass (9 GPT-5.5 findings): restored 1-step-TF prediction loss, pruned `lewm-v3` registry stub, renamed `compile_and_warm`, added collection fail-loud + `--allow-partial`, scoped pytest to `tests/`, full docs refresh. |
| **M5** | (open) | Reward ablation: how much of G1's walking behavior survives with the `tracking_lin_vel` reward zeroed out and only self-supervised prediction + intrinsic-motivation signals driving exploration? First test of the self-learning commitment. (Bridges into Phase 4.) |

---

### Phase 4 — Self-Learning *(current focus after Phase 3 closure)*

Intrinsic motivation (RND / Plan2Explore / empowerment), episodic memory,
long-horizon and sparse-reward tasks. The research question proper begins
here.

Not scoped in detail yet — picked milestone-by-milestone as the Phase 3
library stabilizes into production.

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
7. [ADR 007 — LeWorldModel as the plastic world model](decisions/007-lewm-world-model.md)
8. [ADR 008 — Hierarchical World Models (HWM) as the planning layer](decisions/008-hwm-planning-layer.md)

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
