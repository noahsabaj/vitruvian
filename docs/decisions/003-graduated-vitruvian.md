# ADR 003 — Architecture: Graduated Vitruvian (Frozen Priors + Plastic World Model)

**Status:** Accepted
**Date:** 2026-04-16

## Context

Two legitimate north-stars for embodied AI:

1. **Tabula rasa / "newborn" Vitruvian** — start the agent from random
   weights and let it learn everything from sensorimotor experience,
   driven by intrinsic motivation. Philosophically pure. Slow. Likely
   blocked by sample-complexity walls we do not currently know how to
   climb.
2. **Foundation-model maximalist** — pretrain on video + teleop data,
   fine-tune for the embodiment. Fast, but the philosophically
   uncomfortable consequence is that the agent is not really
   self-learning; it is fine-tuning.

The project's framing dissolves the dichotomy: **biological newborns are
not tabula rasa.** They carry hundreds of millions of years of
evolutionary firmware. The "newborn" metaphor applied honestly means
*plastic cortex on top of evolved substrate*, not *random weights on
empty hardware*.

## Decision

Vitruvian adopts a **graduated architecture** with an explicit partition
between frozen and plastic components:

**Frozen (evolutionary-equivalent):**
- Visual encoder (V-JEPA 2 or DINOv3 candidate — see open questions)
- Motion prior (AMASS-derived, future)
- IMU sensor fusion weights

**Plastic (learned in the agent's lifetime):**
- World model (Dreamer V3 / TD-MPC2 candidate)
- Policy / actor
- Intrinsic motivation signal (RND / Plan2Explore / empowerment candidate)
- Episodic memory

Self-supervised prediction (the world model's loss) is the permanent
learning signal and runs for the agent's entire lifetime. The frozen
components are the perceptual / motor substrate; they are explicitly
*not* "the intelligence."

## Consequences

- We get the sample-efficiency win of pretrained priors without
  abandoning the research commitment to self-learning.
- The partition between frozen and plastic becomes an empirical
  question: which layers should be frozen, which unfrozen, and when?
  This is a legitimate scientific variable rather than a hack.
- Philosophical coherence with biological newborns.
- Risk: creeping reliance on priors for capabilities that *should* be
  learned. Mitigation: maintain an ablation track that periodically
  tests behavior without specific priors, to check whether the plastic
  stack has internalized them.

## Alternatives considered

- **Pure tabula rasa** — rejected as empirically weak and biologically
  inaccurate.
- **Foundation-model fine-tune only** — rejected as not addressing the
  research question (self-learning).

## Open questions

- Which visual encoder: V-JEPA 2 vs DINOv3 vs other?
- Do we ever *un*-freeze a prior, and under what signal?
- Where does language fit in the plastic / frozen split?
