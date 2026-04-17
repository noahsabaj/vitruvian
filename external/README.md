# external/

Vendored external repositories, added as git submodules.

Submodules are added here when we want direct access to upstream assets or
source trees (e.g., `mujoco_menagerie` robot models that we'll be modifying
or comparing against). Python packages consumed as regular dependencies
(`mujoco_playground`, etc.) are installed via `uv` and do *not* live here.

## Planned

- `mujoco_menagerie/` — DeepMind's library of MJCF robot models. Added in M0.2.
