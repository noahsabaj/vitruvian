# ADR 002 — Repo Structure: Monorepo with Vendored Submodules

**Status:** Accepted
**Date:** 2026-04-16

## Context

The project spans simulation, learning algorithms, hardware design
(URDF/MJCF/CAD), and eventually on-robot control. These concerns have
distinct dependencies and will evolve at different rates, but they share
a robot model, evaluation scripts, and configuration.

## Decision

Single `vitruvian` monorepo with internal top-level packages per concern.
External robot-zoo assets (`mujoco_menagerie`) are vendored as git
submodules. External Python packages (`mujoco_playground`, etc.) are
installed via `uv` as dependencies, not vendored, until we outgrow the
upstream API.

Planned layout (indicative; will evolve):

```
vitruvian/
  pyproject.toml
  uv.lock
  src/
    vitruvian/
      sim/         # MuJoCo env wrappers, rendering, randomization
      learn/       # PPO, world models, policies
      robot/       # URDF/MJCF, CAD, kinematic calibration
      control/     # low-level control (future, for hardware)
  docs/
  scripts/
  external/
    mujoco_menagerie/    # submodule
  tests/
```

Hosted on **GitHub, private**, until Milestone 1 is met. Then we revisit
whether to make it public.

## Consequences

- One repo, one `uv` environment, one set of CI concerns.
- `mujoco_menagerie` updates are explicit and auditable via submodule
  pointer commits.
- If upstream `mujoco_playground` changes break us we pin a version; if
  we need to patch it, we fork.
- Risk: the monorepo becomes unwieldy if any one subsystem gets very
  large. Accepted; splitting later is possible.

## Alternatives considered

- **Multiple repos** (`vitruvian-sim`, `vitruvian-learn`, ...) — cleaner
  separation but painful cross-repo versioning. Rejected.
- **Monorepo without submodules** (vendor menagerie by copying) — loses
  upstream tracking. Rejected.
