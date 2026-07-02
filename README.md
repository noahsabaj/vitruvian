# Vitruvian

*A minimum viable humanoid as a substrate for machine intelligence.*

[![Tests](https://github.com/noahsabaj/vitruvian/actions/workflows/test.yml/badge.svg)](https://github.com/noahsabaj/vitruvian/actions/workflows/test.yml)

---

## Status

**Phase 3 world-model stack operational** (2026-04-23). After Phase 1
(G1 walks), Phase 2 (G1 hardening), and the full Phase 3 research arc
(M2 head cam → M3 DINOv3 → M4.1–M4.7 world-model training + planning
+ efficiency refactor), the codebase has been polished into an
installable Python library in M4.8 + M4.9 + M4.9.1:

- `uv pip install -e .` exposes `vit-train`, `vit-plan`, `vit-eval`,
  `vit-collect`, `vit-rollout` as console scripts, each driven by a YAML
  config.
- Single `JEPA` class + `build_jepa` / `load_jepa` registry subsumes
  the milestone-specific composers.
- `MPPIPlanner` with pluggable `CostFn` strategies, `JEPATrainer`
  shared training loop, `EncoderHistory` encode-once ring buffer.
- Vendored LeWM (retired submodule), 88-test CPU suite in under 30s.

Under [ADR 006](docs/decisions/006-g1-reference-body.md), **Unitree G1
is Vitruvian's reference body for the foreseeable future** — the
project builds the software substrate (frozen priors + plastic world
model + self-learning) on top of G1 rather than designing its own
robot.

**M5** (2026-06-30) hardened the world model — a review-driven
correctness pass + a JEPA modernization (SIGReg anti-collapse,
proprio-as-conditioning, DINOv3 kept over V-JEPA) — now retraining from
scratch on cloud hardware as the **baseline**.

Next up: per [ADR 009](docs/decisions/009-thesis-question-milestones.md)
the project pivots from engineering milestones to **thesis-question
milestones** and **decouples the world model from task reward** —
intrinsic cost + intrinsic motivation + goal-conditioned planning, not a
reward-trained policy. The first question (**Q1**): does the planner
beat the policy that trained it, on out-of-distribution goals?

- **Plan:** [`docs/roadmap.md`](docs/roadmap.md)
- **Thesis:** [`docs/thesis.md`](docs/thesis.md)
- **Decisions:** [`docs/decisions/`](docs/decisions/)
- **Session logs:** [`docs/journal/`](docs/journal/)
- **Library API:** `import vitruvian` — see `src/vitruvian/__init__.py`

## Running

Clone (with submodules, for the MuJoCo Menagerie G1 URDF), install:

```bash
git clone --recurse-submodules git@github.com:noahsabaj/vitruvian.git
cd vitruvian
uv sync
```

CLI entry points (each takes a YAML config + optional dotted overrides):

```bash
uv run vit-train   configs/train/jepa_v5.yaml --override trainer.lr=3e-5
uv run vit-plan    configs/plan/forward_walk.yaml
uv run vit-eval    configs/eval/17_run_matrix.yaml
uv run vit-collect configs/collect/diverse.yaml
uv run vit-rollout configs/eval/rollout_accuracy.yaml  # open-loop rollout accuracy (Q1a)
```

Tests:

```bash
uv run pytest            # 88 tests, ~30s, CPU-only (backbones mocked)
```

## License

Apache 2.0. See [`LICENSE`](LICENSE). Vendored upstream code is
credited in [`NOTICE`](NOTICE).
