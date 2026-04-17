# Vitruvian

*A minimum viable humanoid as a substrate for machine intelligence.*

---

## Status

**Phase 1 — Simulation Infrastructure.** M0.1 and M0.2 complete as of
2026-04-16: `uv` project, Python 3.12, `mujoco` + `mujoco-mjx` +
`jax[cuda12]` + `brax` installed, GPU (RTX 4060 Ti) confirmed live,
`mujoco_menagerie` vendored under [`external/`](external/), Unitree G1
parses and steps. Next up: M0.3 (viewer), M0.4 (playground env), M1
(PPO walker).

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
