# Thesis — The Humanoid as a Substrate for Machine Intelligence

*Draft. This document is the project's reason to exist. It will be refined
as the work progresses.*

---

## Claim

The humanoid body is the best general-purpose substrate for machine
intelligence we currently know how to build.

This is not a claim that humanoid robots are the most efficient tool for
any particular task — they are obviously not. A wheeled picker is better
at picking; a quadruped is better at stairs and rubble; an arm on a
workbench is better at assembly. The claim is narrower and more
structural: if the goal is a general-purpose machine that can learn
*anything a human can learn*, inhabiting the environment humans have
already built, the humanoid form is the reference embodiment.

## Why

Three reasons.

**1. The environment is human-shaped.**
Doors, stairs, tools, workbenches, chairs, kitchens, vehicles, terrain,
clothing — the built world is an accumulated affordance library indexed
to human morphology. Any non-humanoid robot that operates in this world
pays a translation cost at every boundary. A humanoid pays no
translation cost.

**2. The data is human-shaped.**
Video of humans doing things is the single largest repository of
embodied behavior in existence. Every frame of it is a potential
demonstration — but only for an agent whose kinematics map to the human
skeleton. The value of this corpus for a wheeled robot is a fraction of
its value for a humanoid.

**3. Intelligence may be embodiment-dependent.**
There is a credible research thesis — not yet proven — that many
cognitive capacities humans treat as abstract (planning, causality,
spatial reasoning, even language) are grounded in sensorimotor
experience. If that is true, the shape of the body is not a neutral
implementation detail but a precondition for the emergence of certain
kinds of mind. A humanoid body is the only body we know of that has
produced a humanoid mind.

## What we reject

We reject **tabula rasa romanticism** — the notion that a machine
intelligence ought to start from nothing. Biological newborns appear to
know nothing and are in fact carrying hundreds of millions of years of
evolutionary firmware: reflex arcs, visual-cortex pre-wiring, attention
biases, motor primitives. Our analogue is pretrained priors: visual
encoders, motion priors, language models. We use them.

We reject **foundation-model maximalism** — the notion that scaling a
single large model on internet data is sufficient and that embodiment is
a deployment problem. The bet here is that certain kinds of intelligence
only emerge from the closed loop of action, sensing, prediction, and
consequence. That loop is the phenomenon we are studying.

## What we commit to

We commit to **self-learning as the plastic core** of the system. The
pretrained priors are the substrate; the lifetime of the agent is
written by its own experience. The world model, the policy, the
intrinsic drives, the episodic memory — these learn continuously from
the agent's own data. This is non-negotiable.

We commit to **simulation first.** Real hardware is the last step, not
the first. Simulation gives us the data rate that biology cannot, and
the reset button that reality does not.

We commit to the **paper trail.** Every architectural decision worth a
six-month-later explanation gets written down, dated, and committed.
Six-month-later us needs to be able to reconstruct the reasoning.

## What we are building

A roughly 65 cm, 20 DoF humanoid, hobby-servo class, 3D-printed
structure, with a simulation-first software stack built around a
continuously-learning world model running on top of frozen perceptual
and motor priors.

In that order: **sim first, robot second, research third.**

## Vocabulary

We prefer **machine intelligence** over *artificial intelligence*. The
word *artificial* implies a stand-in for the real thing. Machine
intelligence is its own thing, with its own substrate, and deserves its
own name.
