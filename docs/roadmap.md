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

### Phase 1 — Simulation Infrastructure

Prove the entire simulation + RL pipeline end-to-end on a known-good
humanoid, on local hardware, reproducibly. No Vitruvian-specific modeling
yet.

| Milestone | Description |
|---|---|
| **M0.1** | `uv` project initialized, Python 3.12, monorepo layout, docs skeleton, first commit, pushed to private GitHub. |
| **M0.2** | `mujoco`, `mujoco-mjx`, `jax[cuda12]`, `brax` installed. `jax.devices()` shows the RTX 4060 Ti. `mujoco_menagerie` added as submodule. |
| **M0.3** | Unitree G1 loads in the MuJoCo viewer, random torques applied, physics confirmed, screen capture saved. |
| **M0.4** | `mujoco_playground` installed, G1 locomotion environment instantiates, random policy rolls out successfully. |
| **M1**   | PPO trains G1 to walk forward at 0.5 m/s. Training reproducible via a single command. wandb run preserved. Video of the trained policy saved. |

**Exit criteria:** M1 met, video committed or linked, wandb dashboard
bookmarked.

---

### Phase 2 — Vitruvian Embodiment

Design the MVH (~20 DoF, ~60-70 cm, hobby-servo class), export to
URDF/MJCF, retrain PPO on the Vitruvian model itself. This is where
mechanical design constraints start feeding back into the policy problem.

Not scoped in detail yet.

---

### Phase 3 — World Models

Replace PPO with Dreamer V3 (or TD-MPC2), mount a simulated head camera,
attach a frozen visual encoder (V-JEPA 2 or DINOv3). This is where the
graduated architecture lands in earnest: frozen evolutionary priors +
plastic world model.

Not scoped in detail yet.

---

### Phase 4 — Self-Learning

Intrinsic motivation (RND / Plan2Explore / empowerment), episodic memory,
long-horizon and sparse-reward tasks. The research question proper begins
here.

Not scoped in detail yet.

---

### Phase 5 — Hardware

3D-printed structure, Dynamixel XL330-class servos, RPi 5 onboard,
sim-to-real via domain randomization.

Deliberately distant. Hardware is last.

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

---

## Open questions carried forward

- Which visual encoder in Phase 3: V-JEPA 2, DINOv3, or other?
- At what point does language enter the stack, and how?
- Which self-learning signal to prioritize in Phase 4: RND vs
  Plan2Explore vs empowerment?
- When (if ever) does a "frozen" prior become unfrozen, and on what signal?
