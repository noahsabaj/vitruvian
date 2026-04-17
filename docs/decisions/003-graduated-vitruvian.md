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

- Which visual encoder(s) — see the 2026-04-16 addendum below.
- Do we ever *un*-freeze a prior, and under what signal?
- Where does language fit in the plastic / frozen split?

---

## Addendum — 2026-04-16 (post-cutoff currency check)

Research review confirmed the thesis of this ADR is intact. The
candidate lists are updated:

### Visual encoders (frozen)

- **V-JEPA 2.1** — arXiv 2603.14482, March 2026 (Mur-Labadia et al.,
  Meta/FAIR). *Current primary candidate for the temporal / video
  prior.* Key additions over V-JEPA 2: dense predictive loss where
  both visible and masked tokens contribute to the training signal,
  deep self-supervision across multiple intermediate encoder layers,
  multi-modal image/video tokenizers, and scaling improvements.
  Crucially for us: **+20 points in real-robot grasping success rate
  vs V-JEPA 2 AC**, which is the benchmark that most closely matches
  Vitruvian's eventual embodied usage pattern. License: **CC-BY-NC-ND
  4.0** (non-commercial, no-derivatives) — see license note at the
  end of this section.
- **V-JEPA 2** — arXiv 2506.09985, June 2025
  (`facebookresearch/vjepa2`). Supersedable baseline kept for
  comparison only.
- **DINOv3** — arXiv 2508.10104, Aug 2025. Released with
  **commercial-license** weights at `facebookresearch/dinov3`.
  Strongest on per-frame dense spatial features. Primary candidate for
  the dense spatial prior.
- **Consensus (per arXiv 2509.21595):** DINOv3 and the V-JEPA lineage
  are complementary, not rivals. Vitruvian's plan is to carry
  **both** — DINOv3 for dense spatial priors (static affordance,
  goal-image matching) and V-JEPA 2.1 for the world-model temporal
  latent — rather than pick one.

**License note on V-JEPA 2.1.** The CC-BY-NC-ND weights restrict
commercial use and derivatives. For our personal-research scope
(Apache 2.0 code + no product shipped) this is fine, but anyone
forking Vitruvian who wants to productize cannot redistribute bundled
V-JEPA 2.1 weights. DINOv3 has no such restriction; if commercial use
ever becomes a concern, the stack should degrade gracefully to
DINOv3-only at the cost of the 20-point grasping gap.

### World model (plastic)

- **Dreamer 4** (arXiv 2509.24527, Sep 2025) — current Hafner-lineage
  SOTA, supersedes Dreamer V3. No official repo yet;
  `nicklashansen/dreamer4` is the best unofficial PyTorch port and
  already targets DMControl continuous control. **Primary Phase 3
  candidate.**
- **TD-MPC2** — still current; worth carrying as a parallel track.
- **Puppeteer** (arXiv 2405.18418, ICLR 2025) — hierarchical TD-MPC2
  variant explicitly built for whole-body humanoid visual control.
  Worth studying once Phase 3 begins.

### Motion priors

- **AMASS** remains the base standard.
- Augment with **LAFAN1** (combat / parkour) and **Motion-X**
  (whole-body incl. hands / face) for 2026 breadth.
- **PULSE** and **PHC** (Luo et al.) as standard latent controllers
  over AMASS.

### Note on RL baseline (affects Phase 1 → Phase 2)

- **FastTD3** (arXiv 2505.22642, May 2025) beats
  PPO/SAC/TD-MPC2/DreamerV3 on Unitree G1/H1 locomotion benchmarks on
  wall-clock. "Sim-to-Real Humanoid Locomotion in 15 Minutes" (arXiv
  2512.01996) is the definitive 2025 result.
- PPO is still the default in `mujoco_playground`, so M1 correctly
  targets it as the reproduce-the-baseline milestone. FastTD3 is
  scheduled in the roadmap as a Phase-2 addition.

All of the above are candidates, not commitments — the thesis of this
ADR (frozen evolutionary-equivalent priors + plastic self-learning
world model) does not change.
