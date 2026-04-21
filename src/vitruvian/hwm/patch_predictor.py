# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""M4.6 — ``PatchARPredictor``: factorized spatio-temporal transformer
that operates on (B, T, N_patches, D) latents.

Each block does:
    1. Temporal multi-head attention (causal, along T) with AdaLN
       conditioning on the per-frame action embedding (from LeWM's
       ``Embedder``) — so the predictor's temporal mixing knows what
       action is being taken.
    2. Temporal MLP with AdaLN.
    3. Spatial multi-head attention (non-causal, along N_patches) —
       unconditioned; lets patches at different body-part locations
       mix within each frame.

This replaces LeWM's ``ARPredictor`` which is temporal-only and assumes
(B, T, D) inputs. We can't reuse that class — it would force the 49
patches to flatten into a single vector, throwing away the spatial
structure we're specifically trying to preserve.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MHA(nn.Module):
    """Standard multi-head self-attention with optional causal mask.

    Written from scratch (rather than reusing LeWM's ``Attention``) to
    avoid the double-LayerNorm that LeWM's ``ConditionalBlock`` and its
    inner ``Attention.norm`` produce, and to keep this block easy to
    reason about.
    """

    def __init__(
        self, dim: int, heads: int, dim_head: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        inner = dim_head * heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.to_out = nn.Linear(inner, dim)
        self.drop = float(dropout)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            t.reshape(B, N, self.heads, -1).transpose(1, 2) for t in qkv
        )  # (B, H, N, d)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.drop if self.training else 0.0,
            is_causal=causal,
        )
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.to_out(out)


class PatchBlock(nn.Module):
    """One transformer block operating on (B, T, N, D) patch-tensors.

    Applies (temporal-attn → temporal-MLP → spatial-attn) in order.
    AdaLN modulation is driven by a per-frame conditioning vector
    ``c`` (typically the action embedding at that frame) and is
    broadcast over the patch axis so all patches in the same frame see
    the same action modulation.

    **LoRA-style shared-trunk AdaLN**: rather than each block having its
    own (SiLU + Linear(dim, 6·dim)) ≈ 394K-param modulation MLP (6 of
    which cost 2.36M params — 26% of the M4.6 predictor), the
    ``PatchARPredictor`` runs a single shared trunk ``SiLU + Linear(dim,
    adaln_rank)`` once per forward and hands each block a per-frame
    low-rank code ``c_trunk`` of shape ``(B, T, adaln_rank)``. The block
    holds only a tiny zero-init head ``Linear(adaln_rank, 6·dim)`` that
    expands the code into its own 6 modulation vectors. With
    ``adaln_rank=128, dim=256`` the per-block AdaLN cost drops from 394K
    to 198K params (× 6 blocks = 2.36M → 1.19M, saving 49%). Zero-init
    on the per-block head preserves the DiT "starts as identity" prior
    exactly.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        adaln_rank: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Temporal (AdaLN-conditioned on action — per-block head only).
        self.norm_t1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn_t = _MHA(dim, heads, dim_head, dropout)
        self.norm_t2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp_t = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        # Per-block AdaLN head: expands the shared low-rank code.
        # Zero-init so the block starts as identity (residual passthrough).
        self.adaln_head = nn.Linear(adaln_rank, 6 * dim, bias=True)
        nn.init.constant_(self.adaln_head.weight, 0)
        nn.init.constant_(self.adaln_head.bias, 0)

        # Spatial (no conditioning).
        self.norm_s = nn.LayerNorm(dim)
        self.attn_s = _MHA(dim, heads, dim_head, dropout)

    def forward(
        self, x: torch.Tensor, c_trunk: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, D) patch-tensor.
            c_trunk: (B, T, adaln_rank) shared AdaLN trunk output from
                the parent ``PatchARPredictor`` — pre-computed once per
                forward and reused across all blocks.
        Returns:
            (B, T, N, D)
        """
        B, T, N, D = x.shape

        mod = self.adaln_head(c_trunk)  # (B, T, 6D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
            6, dim=-1
        )

        def _expand(m):
            return m.unsqueeze(2)  # (B, T, 1, D) broadcasts over N

        # 1. Temporal attention (causal, per-patch-position).
        xt = self.norm_t1(x)
        xt = xt * (1 + _expand(scale_msa)) + _expand(shift_msa)
        # (B, T, N, D) -> (B, N, T, D) -> (B*N, T, D) for per-patch temporal attn.
        xt = xt.permute(0, 2, 1, 3).reshape(B * N, T, D)
        attn_out = self.attn_t(xt, causal=True)
        attn_out = attn_out.reshape(B, N, T, D).permute(0, 2, 1, 3)
        x = x + _expand(gate_msa) * attn_out

        # 2. Temporal MLP (AdaLN-conditioned).
        xm = self.norm_t2(x)
        xm = xm * (1 + _expand(scale_mlp)) + _expand(shift_mlp)
        x = x + _expand(gate_mlp) * self.mlp_t(xm)

        # 3. Spatial attention (non-causal, within-frame).
        xs = self.norm_s(x).reshape(B * T, N, D)
        attn_out = self.attn_s(xs, causal=False).reshape(B, T, N, D)
        x = x + attn_out

        return x


class PatchARPredictor(nn.Module):
    """Factorized spatial+temporal predictor over patch-tensors.

    Shapes:
        input  x: (B, T, N_patches, input_dim)
        input  c: (B, T, hidden_dim)            — action embedding per frame
        output:   (B, T, N_patches, output_dim)

    Applies `input_proj` if input_dim != hidden_dim, then adds learned
    2D positional embedding (spatial + temporal), runs ``depth`` blocks,
    and applies a final LayerNorm and `output_proj`.
    """

    def __init__(
        self,
        *,
        num_frames: int,
        num_patches: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        input_dim: int,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        dim_head: int = 32,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        adaln_rank: int = 128,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        output_dim = output_dim or input_dim

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        # Learned positional embeddings. Kept small at init to avoid
        # swamping the patch features.
        self.pos_spatial = nn.Parameter(
            torch.randn(1, 1, num_patches, hidden_dim) * 0.02
        )
        self.pos_temporal = nn.Parameter(
            torch.randn(1, num_frames, 1, hidden_dim) * 0.02
        )
        self.emb_drop = nn.Dropout(emb_dropout)

        # Shared AdaLN trunk — see PatchBlock docstring for rationale.
        # Feeds every block a pre-computed ``(B, T, adaln_rank)`` code
        # so each block holds only a thin zero-init expansion head.
        self.adaln_trunk = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, adaln_rank),
        )

        self.blocks = nn.ModuleList(
            [
                PatchBlock(
                    hidden_dim, heads, dim_head, mlp_dim, adaln_rank, dropout
                )
                for _ in range(depth)
            ]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

        self.num_frames = int(num_frames)
        self.num_patches = int(num_patches)
        self.hidden_dim = int(hidden_dim)
        self.adaln_rank = int(adaln_rank)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, input_dim)
            c: (B, T, hidden_dim)
        Returns:
            (B, T, N, output_dim)
        """
        assert x.dim() == 4, f"expected (B, T, N, D), got {tuple(x.shape)}"
        x = self.input_proj(x)
        T = x.shape[1]
        N = x.shape[2]
        x = x + self.pos_spatial[:, :, :N] + self.pos_temporal[:, :T]
        x = self.emb_drop(x)
        # Compute shared AdaLN trunk once; all blocks read from it.
        c_trunk = self.adaln_trunk(c)  # (B, T, adaln_rank)
        for blk in self.blocks:
            x = blk(x, c_trunk)
        x = self.norm_out(x)
        return self.output_proj(x)
