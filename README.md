# Vitruvian

*A minimum viable humanoid as a substrate for machine intelligence.*

---

## Status

**Phase 1 closed — G1 walks** (2026-04-16). All five Phase 1
milestones (M0.1–M0.4 + M1) complete; the PPO policy tracks the
commanded joystick velocity on `G1JoystickFlatTerrain` with no falls,
user-confirmed on the final rollout GIF.

Under [ADR 006](docs/decisions/006-g1-reference-body.md), **Unitree G1
is Vitruvian's reference body for the foreseeable future** — the
project builds the software substrate (frozen priors + plastic world
model + self-learning) on top of G1 rather than designing its own
robot.

Next up: **Phase 3** (world models). First milestone is M2 — add a
simulated head camera to G1 so visual observations flow through the
training pipeline.

- **Plan:** [`docs/roadmap.md`](docs/roadmap.md)
- **Thesis:** [`docs/thesis.md`](docs/thesis.md)
- **Decisions:** [`docs/decisions/`](docs/decisions/)
- **Session logs:** [`docs/journal/`](docs/journal/)

## Running

Clone (with submodules), install, prepare the shell:

```bash
git clone --recurse-submodules git@github.com:noahsabaj/vitruvian.git
cd vitruvian
uv sync
source scripts/env-setup.sh    # unsets LD_LIBRARY_PATH (see journal 2026-04-16)
```

## License

Apache 2.0. See [`LICENSE`](LICENSE).
