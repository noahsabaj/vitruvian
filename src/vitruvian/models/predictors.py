# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Predictors — sequence-to-next-step operators over latent embeddings.

Two concrete predictors are exposed:

* :class:`ARPredictor` (re-exported from :mod:`vitruvian.lewm_compat`) —
  temporal-only autoregressive transformer for ``(B, T, D)`` latents;
  used by v3 and v4 composers.
* :class:`PatchARPredictor` — factorized spatial + temporal transformer
  for ``(B, T, N_patches, D)`` latents; used by v5. Carries the
  LoRA-style shared-AdaLN trunk introduced in M4.7.D.

Both conform to the same call contract: ``(x, c) -> x`` where ``c`` is
a per-frame conditioning embedding (typically the action embedding).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vitruvian.lewm_compat import ARPredictor as ARPredictor  # re-export


class _MHA(nn.Module):
    """Standard multi-head self-attention with optional causal mask.

    Written from scratch rather than reusing LeWM's ``Attention`` so this
    block avoids the double-LayerNorm LeWM's ``ConditionalBlock`` + its
    inner ``Attention.norm`` would produce.
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
        )
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.drop if self.training else 0.0,
            is_causal=causal,
        )
        out = out.transpose(1, 2).reshape(B, N, -1)
        projected: torch.Tensor = self.to_out(out)
        return projected


class PatchBlock(nn.Module):
    """One transformer block operating on ``(B, T, N, D)`` patch tensors.

    Applies (temporal-attn → temporal-MLP → spatial-attn) in order.
    AdaLN modulation is driven by a per-frame conditioning vector
    broadcast over the patch axis so all patches in the same frame see
    the same action modulation.

    **LoRA-style shared-trunk AdaLN** (M4.7.D): rather than each block
    owning its own ``SiLU + Linear(dim, 6·dim)`` ≈ 394K-param MLP, the
    parent :class:`PatchARPredictor` runs a single shared trunk
    ``SiLU + Linear(dim, adaln_rank)`` once per forward and hands each
    block a per-frame low-rank code of shape ``(B, T, adaln_rank)``.
    Each block holds only a zero-init head ``Linear(adaln_rank, 6·dim)``.
    At ``adaln_rank=128, dim=256`` the per-block AdaLN cost drops from
    394K → 198K params; across 6 blocks, 2.36M → 1.19M (saving 49%).
    Zero-init on the head preserves the DiT "starts as identity" prior.
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
        self.adaln_head = nn.Linear(adaln_rank, 6 * dim, bias=True)
        nn.init.constant_(self.adaln_head.weight, 0)
        nn.init.constant_(self.adaln_head.bias, 0)

        self.norm_s = nn.LayerNorm(dim)
        self.attn_s = _MHA(dim, heads, dim_head, dropout)

    def forward(self, x: torch.Tensor, c_trunk: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape

        mod = self.adaln_head(c_trunk)  # (B, T, 6D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
            6, dim=-1
        )

        def _expand(m: torch.Tensor) -> torch.Tensor:
            return m.unsqueeze(2)

        xt = self.norm_t1(x)
        xt = xt * (1 + _expand(scale_msa)) + _expand(shift_msa)
        xt = xt.permute(0, 2, 1, 3).reshape(B * N, T, D)
        attn_out = self.attn_t(xt, causal=True)
        attn_out = attn_out.reshape(B, N, T, D).permute(0, 2, 1, 3)
        x = x + _expand(gate_msa) * attn_out

        xm = self.norm_t2(x)
        xm = xm * (1 + _expand(scale_mlp)) + _expand(shift_mlp)
        x = x + _expand(gate_mlp) * self.mlp_t(xm)

        xs = self.norm_s(x).reshape(B * T, N, D)
        attn_out = self.attn_s(xs, causal=False).reshape(B, T, N, D)
        x = x + attn_out

        return x


class PatchARPredictor(nn.Module):
    """Factorized spatial + temporal predictor over patch tensors.

    Shapes:
        input  ``x``: ``(B, T, N_patches, input_dim)``
        input  ``c``: ``(B, T, hidden_dim)`` — action embedding per frame
        output:      ``(B, T, N_patches, output_dim)``

    Applies ``input_proj`` if ``input_dim != hidden_dim``, then adds a
    learned 2D positional embedding (spatial + temporal), runs ``depth``
    :class:`PatchBlock` layers, and finishes with a LayerNorm and
    ``output_proj``.
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

        self.pos_spatial = nn.Parameter(
            torch.randn(1, 1, num_patches, hidden_dim) * 0.02
        )
        self.pos_temporal = nn.Parameter(
            torch.randn(1, num_frames, 1, hidden_dim) * 0.02
        )
        self.emb_drop = nn.Dropout(emb_dropout)

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
        assert x.dim() == 4, f"expected (B, T, N, D), got {tuple(x.shape)}"
        x = self.input_proj(x)
        T = x.shape[1]
        N = x.shape[2]
        x = x + self.pos_spatial[:, :, :N] + self.pos_temporal[:, :T]
        x = self.emb_drop(x)
        c_trunk = self.adaln_trunk(c)
        for blk in self.blocks:
            x = blk(x, c_trunk)
        x = self.norm_out(x)
        out: torch.Tensor = self.output_proj(x)
        return out


class _SpatialBlock(nn.Module):
    """AdaLN-modulated spatial transformer block over ``(B, K, N, D)``.

    One *independent* prediction per horizon ``k`` — spatial self-attention
    over the ``N`` patches only, no cross-horizon (temporal) attention — so
    Fast-LeWM's prefix predictions do not chain. Reuses the shared low-rank
    AdaLN code (``c_trunk``, one per horizon) with a zero-init head (DiT
    identity prior).
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
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn_s = _MHA(dim, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        self.adaln_head = nn.Linear(adaln_rank, 6 * dim, bias=True)
        nn.init.constant_(self.adaln_head.weight, 0)
        nn.init.constant_(self.adaln_head.bias, 0)

    def forward(self, x: torch.Tensor, c_trunk: torch.Tensor) -> torch.Tensor:
        B, K, N, D = x.shape
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaln_head(
            c_trunk
        ).chunk(6, dim=-1)  # each (B, K, D)

        def _e(m: torch.Tensor) -> torch.Tensor:
            return m.unsqueeze(2)  # (B, K, 1, D)

        xs = self.norm1(x)
        xs = xs * (1 + _e(scale1)) + _e(shift1)
        attn = self.attn_s(xs.reshape(B * K, N, D), causal=False).reshape(
            B, K, N, D
        )
        x = x + _e(gate1) * attn

        xm = self.norm2(x)
        xm = xm * (1 + _e(scale2)) + _e(shift2)
        x = x + _e(gate2) * self.mlp(xm)
        return x


class PrefixPatchPredictor(nn.Module):
    """Fast-LeWM (arXiv:2606.26217) action-prefix parallel predictor.

    Given the anchor (current) patch latent ``z_t`` and per-step action
    embeddings ``a_t..a_{t+H-1}``, predicts ``ẑ_{t+1..t+H}`` in **one pass** —
    each horizon anchored on ``z_t`` and conditioned on the action *prefix* up
    to that horizon (via a causal action-prefix encoder). Predictions never
    chain, so there is no compounding rollout error and all horizons compute in
    parallel (the M6 fix for the horizon-error growth Q1a measured).

    Shapes::

        anchor  : (B, N, input_dim)      anchor patch latent
        act_emb : (B, H, hidden_dim)     per-step action embeddings
        output  : (B, H, N, output_dim)  ẑ_{t+1..t+H}
    """

    def __init__(
        self,
        *,
        num_patches: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        input_dim: int,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        dim_head: int = 32,
        prefix_depth: int = 3,
        dropout: float = 0.0,
        adaln_rank: int = 128,
        max_horizon: int = 8,
    ) -> None:
        super().__init__()
        hidden = hidden_dim or input_dim
        out = output_dim or input_dim

        self.input_proj = (
            nn.Linear(input_dim, hidden) if input_dim != hidden else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(hidden, out) if hidden != out else nn.Identity()
        )
        self.pos_spatial = nn.Parameter(torch.randn(1, num_patches, hidden) * 0.02)

        # Action-prefix causal encoder over [state_token, a_0..a_{H-1}].
        self.state_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.prefix_pos = nn.Parameter(
            torch.randn(1, max_horizon + 1, hidden) * 0.02
        )
        self.prefix_ln = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(prefix_depth)]
        )
        self.prefix_attn = nn.ModuleList(
            [_MHA(hidden, heads, dim_head, dropout) for _ in range(prefix_depth)]
        )
        self.prefix_mlp_ln = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(prefix_depth)]
        )
        self.prefix_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, hidden)
                )
                for _ in range(prefix_depth)
            ]
        )
        self.prefix_norm = nn.LayerNorm(hidden)

        # Parallel spatial predictor.
        self.adaln_trunk = nn.Sequential(nn.SiLU(), nn.Linear(hidden, adaln_rank))
        self.blocks = nn.ModuleList(
            [
                _SpatialBlock(hidden, heads, dim_head, mlp_dim, adaln_rank, dropout)
                for _ in range(depth)
            ]
        )
        self.norm_out = nn.LayerNorm(hidden)

        self.num_patches = int(num_patches)
        self.hidden_dim = int(hidden)
        self.max_horizon = int(max_horizon)
        self.adaln_rank = int(adaln_rank)

    def _encode_prefixes(
        self, state_tok: torch.Tensor, act_emb: torch.Tensor
    ) -> torch.Tensor:
        """[state, a_0..a_{H-1}] --causal--> prefix tokens (B, H, hidden), where
        token k summarizes only a_0..a_{k-1} (plus the state token)."""
        H = act_emb.shape[1]
        toks = torch.cat([state_tok.unsqueeze(1), act_emb], dim=1)  # (B, H+1, h)
        toks = toks + self.prefix_pos[:, : H + 1]
        for ln, attn, mln, mlp in zip(
            self.prefix_ln, self.prefix_attn, self.prefix_mlp_ln, self.prefix_mlp
        ):
            toks = toks + attn(ln(toks), causal=True)
            toks = toks + mlp(mln(toks))
        return self.prefix_norm(toks[:, 1:])  # drop the 0-th (state) output

    def forward(self, anchor: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        assert anchor.dim() == 3, f"anchor (B, N, D), got {tuple(anchor.shape)}"
        B, N, _ = anchor.shape
        H = act_emb.shape[1]
        if H > self.max_horizon:
            raise ValueError(f"horizon {H} exceeds max_horizon {self.max_horizon}")
        z = self.input_proj(anchor) + self.pos_spatial[:, :N]  # (B, N, hidden)
        state_tok = self.state_mlp(z.mean(dim=1))              # (B, hidden)
        prefix = self._encode_prefixes(state_tok, act_emb)     # (B, H, hidden)
        c_trunk = self.adaln_trunk(prefix)                     # (B, H, adaln_rank)
        x = z.unsqueeze(1).expand(B, H, N, self.hidden_dim).contiguous()
        for blk in self.blocks:
            x = blk(x, c_trunk)
        x = self.norm_out(x)
        out: torch.Tensor = self.output_proj(x)
        return out  # (B, H, N, output_dim)


__all__ = [
    "ARPredictor",
    "PatchARPredictor",
    "PatchBlock",
    "PrefixPatchPredictor",
]
