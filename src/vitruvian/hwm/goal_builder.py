# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Macro-latent → primitive-action decoder (nearest-neighbour retrieval).

HWM's paper (arXiv:2604.03208 §2.3) decodes planned macro latents by
running a second MPPI loop over primitive actions to drive the state
toward the predicted subgoal latent. We start with a cheaper, simpler
decoder: pre-encode every macro chunk in the expert dataset, then
retrieve the expert primitives whose macro-latent is closest in L2 to
the planned macro latent.

See docs/decisions/008-hwm-planning-layer.md (§ Macro-to-primitive
decoding) for the fallback ladder.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from .action_codec import MacroActionEncoder


class MacroNNRetriever:
    """Build a k-NN index over expert macros, then retrieve primitives.

    Given an expert dataset in HDF5 with ``action (N_total, 29)``,
    ``ep_offset``, ``ep_len``, we:

        1. Chunk each episode into non-overlapping macros of
           ``step_skip`` primitives.
        2. Encode every chunk through ``MacroActionEncoder``.
        3. Store (macro_latent, primitives) pairs in memory for
           nearest-neighbour lookup.

    At plan time, ``retrieve(latent)`` returns the primitives of the
    macro whose latent has the smallest L2 distance to ``latent``.
    """

    def __init__(
        self,
        h5_path: str | Path,
        action_encoder: MacroActionEncoder,
        device: str = "cuda",
        batch_size: int = 256,
    ) -> None:
        self.h5_path = str(h5_path)
        self.step_skip = int(action_encoder.step_skip)
        self.action_dim = int(action_encoder.action_dim)
        self.device = device
        self.encoder = action_encoder.to(device).eval()

        with h5py.File(self.h5_path, "r") as f:
            ep_offset = f["ep_offset"][:].astype(np.int64)
            ep_len = f["ep_len"][:].astype(np.int64)
            action = f["action"][:].astype(np.float32)

        chunks: list[np.ndarray] = []
        for off, L in zip(ep_offset, ep_len):
            n_macros = int(L) // self.step_skip
            if n_macros <= 0:
                continue
            seg = action[off : off + n_macros * self.step_skip]
            seg = seg.reshape(n_macros, self.step_skip, self.action_dim)
            chunks.append(seg)
        if not chunks:
            raise RuntimeError(
                f"No macro chunks could be formed from {self.h5_path} — "
                f"check ep_len vs step_skip={self.step_skip}."
            )
        self.primitives = torch.from_numpy(np.concatenate(chunks, axis=0))
        # (N_macros, step_skip, action_dim) — kept on CPU; small.

        # Encode in batches on the chosen device, keep the index on CPU
        # (floats are small: ~N_macros × 32 × 4 bytes; lookup is fast).
        latents: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, self.primitives.shape[0], batch_size):
                b = self.primitives[i : i + batch_size].to(device)
                # MacroActionEncoder expects (B, T_macro, step_skip, A)
                z = self.encoder(b.unsqueeze(1)).squeeze(1)
                latents.append(z.cpu())
        self.latents = torch.cat(latents, dim=0)  # (N_macros, macro_act_dim)

    def __len__(self) -> int:
        return int(self.primitives.shape[0])

    def retrieve(
        self, latent: torch.Tensor, k: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the primitives of the k nearest expert macros.

        Args:
            latent: (macro_act_dim,) query latent.
            k: number of neighbours to return (stacked on dim 0).
        Returns:
            primitives: (k, step_skip, action_dim) float32 CPU tensor.
            distances:  (k,) L2 distances.
        """
        q = latent.detach().cpu().float().view(1, -1)
        d = torch.cdist(q, self.latents).squeeze(0)  # (N_macros,)
        top = torch.topk(d, k=k, largest=False)
        idx = top.indices
        return self.primitives[idx], top.values

    def retrieve_first(self, latent: torch.Tensor) -> torch.Tensor:
        """Convenience: primitives of the single closest neighbour,
        shape (step_skip, action_dim)."""
        prim, _ = self.retrieve(latent, k=1)
        return prim[0]
