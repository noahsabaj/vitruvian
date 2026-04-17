# ADR 006 — Unitree G1 as Vitruvian's Reference Body

**Status:** Accepted
**Date:** 2026-04-16

## Context

The project's original Phase 2 was to **design the MVH** (minimum
viable humanoid) — a ~65 cm, 20 DoF, hobby-servo-class, 3D-printed
robot — and port its URDF into simulation. Phase 5 then built the real
hardware.

Two reasons to revisit:

1. **CAD is not the user's strength.** The mechanical design work
   required for a good MVH is a full separate project, comparable in
   scope to the software substrate we are actually here to build.
   Attempting it in parallel with the research agenda is how the
   research agenda stalls.
2. **Open and commercial humanoid platforms matured through 2025-2026.**
   `mujoco_menagerie` now ships production-grade MJCFs for Unitree G1,
   Unitree H1, Booster T1, Apptronik Apollo, Fourier N1, PNDbotics Adam
   Lite, Berkeley Humanoid Lite, Robotis OP3, and ToddlerBot. Serious
   embodied-AI labs publish on shared reference platforms rather than
   custom hardware — GR00T (NVIDIA, GR00T N1–N1.7), Pi0 / π*0.6
   (Physical Intelligence), Helix 02 (Figure), Gemini Robotics-ER 1.6
   (DeepMind) all run against commercial/open humanoids.

Phase 1 proved the stack works end-to-end on G1: PPO trained to a
walking policy (eval_reward +3.70, user-confirmed walking with no
falls) reproducibly on our local RTX 4060 Ti.

## Decision

**Unitree G1 is Vitruvian's reference body for the foreseeable future.**

Concretely:

- All research, training, and evaluation happens on
  `external/mujoco_menagerie/unitree_g1/` and its playground envs
  (`G1JoystickFlatTerrain`, `G1JoystickRoughTerrain`, and any
  push-recovery / command-tracking successors).
- **Phase 2** (original: *Vitruvian Embodiment — design the MVH*) is
  **deferred indefinitely.** We revisit only if (a) a credible
  universal open-humanoid design is published that matches the
  Vitruvian target envelope, or (b) the user's CAD situation changes.
- **Phase 5** (original: *Hardware*) is similarly deferred. A
  real-hardware build presupposes a real-hardware design; we don't
  have one.
- The **thesis of the project is unchanged.** Humanoid as machine-
  intelligence substrate, graduated architecture with frozen priors
  and plastic world model, self-learning as the plastic core — all
  stand. G1 is an *instance* of "humanoid body," and the research
  question does not require us to have built the instance ourselves.

The project name stays. *Vitruvian* names the thesis — the humanoid
form as the reference embodiment for machine intelligence. The figure
is inscribed in whichever humanoid-shaped circle we happen to have.
For now, that's G1.

## Consequences

- **No mechanical-engineering bottleneck.** Every week of research
  progress we make is a week of CAD work we are not doing. This is
  a straight multiplier on research throughput.
- **Faster route to research-grade contributions.** The world-model
  stack, the frozen encoders, Dreamer 4, the self-learning loop —
  all can land on G1 without waiting for hardware.
- **Phase 3 onward becomes the project.** World models (Phase 3) and
  self-learning (Phase 4) were always where the research lived; now
  they are explicitly the critical path.
- **Portability is now an open research question, not a given.** A
  policy trained on G1 does not automatically transfer to other
  morphologies. This is a known open problem (cross-embodiment
  transfer) and not specific to us. The *methods* we develop — world
  models, frozen priors, intrinsic motivation — should transfer; the
  specific G1 weights won't.
- **Hardware optionality stays open.** If a good open universal
  humanoid emerges, or if we find a mechanical collaborator later, we
  port the stack. This ADR makes hardware *contingent*, not
  *impossible*.

## Alternatives Considered

- **Original plan: design our own MVH.** Rejected on the CAD-
  bottleneck grounds above.
- **ToddlerBot as Vitruvian stand-in.** Rejected for now: G1 already
  works end-to-end on our stack, and switching would be rework for
  minimal gain. ToddlerBot stays on the books as the primary
  hardware design reference in [ADR 003](003-graduated-vitruvian.md)
  for whenever (if ever) Phase 5 is revisited.
- **Wait until a universal humanoid is published before doing any
  research.** Rejected: the research can continue in parallel on G1
  indefinitely without waiting.

## Open Questions

- At what point does **cross-embodiment transfer** enter the research
  agenda explicitly? A method that works on G1 *and* ToddlerBot
  *and* (eventually) a Vitruvian is a stronger result than a
  G1-only method. Probably matters most in Phase 4 (self-learning),
  when policies are expected to generalize.
- If the `mujoco_menagerie` / HF community publishes a "universal
  hobby humanoid" CAD / URDF under a permissive license before
  Phase 5 would have started, does that flip the deferral? (Probably
  yes.)
