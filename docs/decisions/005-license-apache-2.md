# ADR 005 — License: Apache 2.0

**Status:** Accepted
**Date:** 2026-04-16

## Context

The project is private for now but likely to be opened up at some
point. License choice is most cheaply made before any code is written —
so we make it now and stamp every source file from day 1.

## Decision

The project is licensed under the **Apache License 2.0.**

Source files will carry an SPDX header:

```
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
```

Copyright holder string: `The Vitruvian Authors`.

## Consequences

- Permissive: commercial use, modification, distribution, private use
  all allowed.
- **Patent grant** is explicit. This matters in robotics, where patents
  are common.
- Compatible with the dominant licenses in the robotics / RL ecosystem
  (MuJoCo, JAX, most HuggingFace code, most DeepMind code).
- Every new doc / ADR is implicitly covered by the repo license.

## Alternatives considered

- **MIT** — simpler, widely used, no explicit patent grant. Rejected in
  favor of Apache 2.0's patent provision given the hardware angle.
- **GPL / AGPL** — viral. Rejected; we want permissive reuse.
- **Proprietary / no license** — rejected; default copyright leaves
  collaborators in limbo.
