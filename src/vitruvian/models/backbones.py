# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Backbones — frozen visual encoders exposed via a tiny duck-typed
:class:`Backbone` protocol.

Every backbone we use returns a float32 embedding for a batch of
``(B, T, 3, H, W)`` pixels and advertises an ``output_dim``. Patch-shape
backbones additionally expose ``n_patches`` so the predictor knows how
many spatial tokens to allocate positional embeddings for.

Three concrete implementations live here (previously scattered across
three files in the legacy ``vitruvian.hwm`` package):

* :class:`DINOv3ClsBackbone` — frozen DINOv3 ViT-B/16, returns the
  ``(B, T, 768)`` CLS token per frame. Used by v4 (Chinchilla-scaled
  Terver-recipe) composer.
* :class:`DINOv3PatchBackbone` — same DINOv3 weights, returns a
  stride-subsampled ``(B, T, N, 768)`` patch grid (default 7×7 = 49
  tokens for the 8 GB VRAM budget). Used by v5.
* :class:`LeWMBackbone` — thin wrapper over a trained LeWM-shaped JEPA
  (ViT-tiny 192-D CLS); kept so legacy v3 checkpoints still load.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
import torchvision.transforms.v2 as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_DINOV3_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"


@runtime_checkable
class Backbone(Protocol):
    """Duck-typed protocol every visual backbone implements."""

    output_dim: int

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, T, 3, H, W)`` to ``(B, T, ..., output_dim)`` float32."""
        ...


# --------------------------------------------------------------------------
# DINOv3 CLS
# --------------------------------------------------------------------------


class DINOv3ClsBackbone(nn.Module):
    """Frozen DINOv3 ViT-B/16 CLS-token encoder.

    Input:  ``pixels (B, T, 3, H, W)`` uint8 or float32 in ``[0, 1]``.
    Output: ``emb (B, T, D_out)`` float32 (cast back from FP16 internal).

    The internal DINOv3 runs in FP16 for VRAM headroom (~500 MB saved vs
    FP32); the CLS output is cast to float32 before returning so it can
    participate in autograd for the downstream trainable modules.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_DINOV3_ID,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoModel  # lazy

        self.model_id = model_id
        self.dtype = dtype
        self.device_str = device
        self.frozen = freeze

        self.dinov3 = AutoModel.from_pretrained(model_id, dtype=dtype).to(device)
        if freeze:
            for p in self.dinov3.parameters():
                p.requires_grad_(False)
            self.dinov3.eval()

        hidden = int(self.dinov3.config.hidden_size)
        self.output_dim = hidden

        self._normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device, dtype=dtype)
            out = self.dinov3(pixel_values=dummy)
            assert out.last_hidden_state.shape[-1] == hidden, (
                f"DINOv3 last_hidden_state last dim {out.last_hidden_state.shape[-1]} "
                f"!= config.hidden_size {hidden}"
            )

    def _prep(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.dtype == torch.uint8:
            pixels = pixels.float() / 255.0
        elif pixels.dtype != torch.float32 and pixels.dtype != torch.float16:
            pixels = pixels.float()
        pixels = self._normalize(pixels)
        return pixels.to(self.dtype)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode pixel frames → flat CLS embeddings.

        Args:
            pixels: ``(B, T, 3, H, W)``; ``T`` may be 1.
        Returns:
            ``emb (B, T, D_out)`` float32.
        """
        assert pixels.dim() == 5, (
            f"expected (B, T, 3, H, W), got {tuple(pixels.shape)}"
        )
        B, T = pixels.shape[0], pixels.shape[1]
        pixels_flat = pixels.reshape(B * T, *pixels.shape[2:])
        pixels_flat = self._prep(pixels_flat)
        if self.frozen:
            with torch.no_grad():
                out = self.dinov3(pixel_values=pixels_flat)
        else:
            out = self.dinov3(pixel_values=pixels_flat)
        cls = out.last_hidden_state[:, 0, :]  # (B*T, D)
        cls = cls.float().reshape(B, T, -1)
        return cls

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.encode(pixels)


# --------------------------------------------------------------------------
# DINOv3 patches
# --------------------------------------------------------------------------


class DINOv3PatchBackbone(nn.Module):
    """Frozen DINOv3 ViT-B/16 returning stride-subsampled patch tokens.

    Input:  ``pixels (B, T, 3, H, W)`` uint8 or float (any dtype).
    Output: ``emb (B, T, N_patches, D_out)`` float32
            where ``N_patches = ((H/16)//stride)**2`` (49 for H=224,
            stride=2) and ``D_out = encoder.hidden_size`` (768 for
            ViT-B/16).

    The encoder runs in FP16 for VRAM; the returned tensor is cast to
    float32 so downstream modules (trainable projection, predictor) can
    hold autograd state without FP16 precision loss.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_DINOV3_ID,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        freeze: bool = True,
        spatial_stride: int = 2,
        lazy: bool = False,
    ) -> None:
        """
        Args:
            lazy: If True, skip loading the DINOv3 weights into VRAM at
                construction. Use when training reads a precomputed
                patch cache — the 344 MB of weights would otherwise
                waste VRAM that the trainable predictor needs. The
                geometry fields (``output_dim``, ``grid_side``,
                ``n_patches``) are set from ViT-B/16 defaults so
                downstream shape checks still work. Call
                :meth:`load_eagerly` to hydrate the weights later
                (needed for plan-time encoding).
        """
        super().__init__()

        self.model_id = model_id
        self.dtype = dtype
        self.device_str = device
        self.frozen = freeze
        self.spatial_stride = int(spatial_stride)
        self._normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.dinov3: nn.Module | None = None

        if lazy:
            self.output_dim = 768
            self._n_skip = 5  # CLS + 4 registers
            self.grid_side_full = 14
            self.grid_side = 14 // self.spatial_stride
            self.n_patches = self.grid_side * self.grid_side
            return

        self.load_eagerly()

    def load_eagerly(self) -> None:
        """Load the DINOv3 weights into VRAM. Idempotent."""
        if self.dinov3 is not None:
            return
        from transformers import AutoModel

        self.dinov3 = AutoModel.from_pretrained(
            self.model_id, dtype=self.dtype
        ).to(self.device_str)
        if self.frozen:
            for p in self.dinov3.parameters():
                p.requires_grad_(False)
            self.dinov3.eval()

        hidden = int(self.dinov3.config.hidden_size)
        self.output_dim = hidden
        n_reg = int(getattr(self.dinov3.config, "num_register_tokens", 4))
        self._n_skip = 1 + n_reg

        with torch.no_grad():
            dummy = torch.zeros(
                1, 3, 224, 224, device=self.device_str, dtype=self.dtype
            )
            out = self.dinov3(pixel_values=dummy)
            seq_len = int(out.last_hidden_state.shape[1])
            n_patches_full = seq_len - self._n_skip
            side = int(n_patches_full**0.5)
            assert side * side == n_patches_full, (
                f"DINOv3 output {n_patches_full} patches is not a square grid"
            )
        self.grid_side_full = side
        self.grid_side = side // self.spatial_stride
        self.n_patches = self.grid_side * self.grid_side

    def _prep(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.dtype == torch.uint8:
            pixels = pixels.float() / 255.0
        elif pixels.dtype not in (torch.float32, torch.float16):
            pixels = pixels.float()
        pixels = self._normalize(pixels)
        return pixels.to(self.dtype)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode pixel frames → stride-subsampled patch tokens.

        Args:
            pixels: ``(B, T, 3, H, W)`` where ``H=W=224`` for standard
                DINOv3.
        Returns:
            ``emb (B, T, N_patches, D_out)`` float32.
        """
        if self.dinov3 is None:
            self.load_eagerly()
        assert pixels.dim() == 5, (
            f"expected (B, T, 3, H, W), got {tuple(pixels.shape)}"
        )
        B, T = pixels.shape[0], pixels.shape[1]
        pixels_flat = pixels.reshape(B * T, *pixels.shape[2:])
        pixels_flat = self._prep(pixels_flat)
        if self.frozen:
            with torch.no_grad():
                out = self.dinov3(pixel_values=pixels_flat)
        else:
            out = self.dinov3(pixel_values=pixels_flat)
        patches = out.last_hidden_state[:, self._n_skip :, :]  # (B*T, G*G, D)
        G = self.grid_side_full
        patches = patches.reshape(B * T, G, G, -1)
        if self.spatial_stride > 1:
            patches = patches[:, :: self.spatial_stride, :: self.spatial_stride, :]
        patches = patches.reshape(B * T, self.n_patches, -1)
        patches = patches.float().reshape(B, T, self.n_patches, -1)
        return patches

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.encode(pixels)


# --------------------------------------------------------------------------
# LeWM-compatible backbone (legacy v3 checkpoints)
# --------------------------------------------------------------------------


class LeWMBackbone(nn.Module):
    """Thin wrapper over a trained LeWM-shaped JEPA encoder.

    Input:  ``pixels (B, T, 3, H, W)`` float in [0, 1] (uint8 also OK).
    Output: ``emb (B, T, D)`` float32, ``D = encoder hidden_size``
            (192 for ViT-tiny).
    """

    def __init__(self, jepa: nn.Module, freeze: bool = True) -> None:
        super().__init__()
        self.jepa = jepa
        self.frozen = freeze
        if freeze:
            for p in self.jepa.parameters():
                p.requires_grad_(False)
            self.jepa.eval()

        self._normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        with torch.no_grad():
            dummy_dev = next(self.jepa.parameters()).device
            dummy = torch.zeros(1, 1, 3, 224, 224, device=dummy_dev)
            info = self.jepa.encode({"pixels": self._prep(dummy)})
            emb = info["emb"]
        assert emb.dim() == 3, f"unexpected emb shape {tuple(emb.shape)}"
        self.output_dim = int(emb.shape[-1])

    def _prep(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.dtype == torch.uint8:
            pixels = pixels.float() / 255.0
        elif pixels.dtype != torch.float32:
            pixels = pixels.float()
        return self._normalize(pixels)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = self._prep(pixels)
        if self.frozen:
            with torch.no_grad():
                info = self.jepa.encode({"pixels": pixels})
        else:
            info = self.jepa.encode({"pixels": pixels})
        return info["emb"]

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.encode(pixels)


def load_lewm_jepa_from_checkpoint(
    ckpt_path: str,
    lewm_repo_path: str | None = None,
    device: str = "cuda",
) -> nn.Module:
    """Rebuild a standalone LeWM-shaped JEPA from a Lightning checkpoint.

    Dedicated loader for the legacy v3 checkpoints that predate the
    unified schema. M4.5+ v4/v5 checkpoints should go through
    :func:`vitruvian.models.load_jepa` instead — it handles v4/v5
    schemas with in-memory migration.

    Uses the vendored :mod:`vitruvian.lewm_compat` classes — no need
    for the ``external/le-wm/`` submodule to be on ``sys.path``.
    """
    del lewm_repo_path  # legacy arg — kept for API compatibility.

    from transformers import ViTConfig, ViTModel

    from vitruvian.lewm_compat import (
        MLP as LeWMMLP,
        ARPredictor as LeWMARPredictor,
        Embedder as LeWMEmbedder,
        JEPA as LeWMJEPA,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]

    def _vit_tiny_config() -> ViTConfig:
        return ViTConfig(
            hidden_size=192,
            num_hidden_layers=12,
            num_attention_heads=3,
            intermediate_size=768,
            image_size=224,
            patch_size=14,
        )

    enc_cfg = _vit_tiny_config()
    encoder = ViTModel(enc_cfg, add_pooling_layer=False)

    ae_keys = [k for k in state if "action_encoder.patch_embed.weight" in k]
    if ae_keys:
        ae_weight = state[ae_keys[0]]
        action_input_dim = int(ae_weight.shape[1])
        action_smoothed_dim = int(ae_weight.shape[0])
    else:
        action_input_dim = 29
        action_smoothed_dim = 10

    emb_dim_keys = [k for k in state if "action_encoder.embed.2.weight" in k]
    if emb_dim_keys:
        action_emb_dim = int(state[emb_dim_keys[0]].shape[0])
    else:
        action_emb_dim = 10

    action_encoder = LeWMEmbedder(
        input_dim=action_input_dim,
        smoothed_dim=action_smoothed_dim,
        emb_dim=action_emb_dim,
    )
    projector = LeWMMLP(
        input_dim=enc_cfg.hidden_size,
        output_dim=enc_cfg.hidden_size,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )
    pred_proj = LeWMMLP(
        input_dim=enc_cfg.hidden_size,
        output_dim=enc_cfg.hidden_size,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    stripped = {
        k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")
    }

    pos_key = "predictor.pos_embedding"
    if pos_key in stripped:
        pos = stripped[pos_key]
        ar_num_frames = int(pos.shape[1])
        ar_input_dim = int(pos.shape[2])
    else:
        ar_num_frames, ar_input_dim = 3, enc_cfg.hidden_size
    predictor = LeWMARPredictor(
        num_frames=ar_num_frames,
        depth=6,
        heads=16,
        mlp_dim=2048,
        input_dim=ar_input_dim,
        hidden_dim=ar_input_dim,
        output_dim=ar_input_dim,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.0,
    )

    proprio_encoder = None
    prop_w0 = stripped.get("proprio_encoder.net.0.weight")
    prop_w3 = stripped.get("proprio_encoder.net.3.weight")
    if prop_w0 is not None and prop_w3 is not None:
        proprio_in = int(prop_w0.shape[1])
        proprio_hidden = int(prop_w0.shape[0])
        proprio_out = int(prop_w3.shape[0])
        proprio_encoder = LeWMMLP(
            input_dim=proprio_in,
            output_dim=proprio_out,
            hidden_dim=proprio_hidden,
            norm_fn=None,
        )

    jepa_model = LeWMJEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
        proprio_encoder=proprio_encoder,
    )

    missing, unexpected = jepa_model.load_state_dict(stripped, strict=False)
    if missing:
        raise RuntimeError(
            f"load_state_dict: missing keys: {missing[:5]}... "
            f"(total {len(missing)})"
        )
    if unexpected:
        raise RuntimeError(
            f"load_state_dict: unexpected keys: {unexpected[:5]}... "
            f"(total {len(unexpected)})"
        )

    jepa_model = jepa_model.to(device).eval()
    return jepa_model


__all__ = [
    "Backbone",
    "DEFAULT_DINOV3_ID",
    "DINOv3ClsBackbone",
    "DINOv3PatchBackbone",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "LeWMBackbone",
    "load_lewm_jepa_from_checkpoint",
]
