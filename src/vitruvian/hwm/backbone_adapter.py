# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""LeWM backbone adapter.

Wraps a trained LeWM ``jepa.JEPA`` instance to expose a minimal flat
encoder API for HWM's hierarchical planner. Returns a single flat
CLS embedding per frame — no ``BackboneOutput`` namedtuple, no
``output_obs_dim``/``output_proprio_dim`` tuples, no location
extraction. See docs/decisions/008-hwm-planning-layer.md for the
architectural mismatch justifying this.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms.v2 as T

# ImageNet normalization — LeWM's encoder expects its frozen ViT to
# receive ImageNet-normalized floats per stable_pretraining.data
# dataset_stats.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LeWMBackboneAdapter(nn.Module):
    """Thin wrapper over LeWM's JEPA encoder.

    Input:  pixels (B, T, 3, H, W) float in [0, 1]  (or uint8; handled)
    Output: emb    (B, T, D)        float32  where D = encoder hidden
    """

    def __init__(self, jepa: nn.Module, freeze: bool = True) -> None:
        super().__init__()
        self.jepa = jepa
        self.frozen = freeze
        if freeze:
            for p in self.jepa.parameters():
                p.requires_grad_(False)
            self.jepa.eval()

        # Try to infer output dim from the encoder. ViT-tiny gives 192.
        # Probe with a dummy 1-frame batch. Done at construction so a
        # wrong wrapper is detected early.
        self._normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        with torch.no_grad():
            dummy_dev = next(self.jepa.parameters()).device
            dummy = torch.zeros(1, 1, 3, 224, 224, device=dummy_dev)
            info = self.jepa.encode({"pixels": self._prep(dummy)})
            emb = info["emb"]
        assert emb.dim() == 3, f"unexpected emb shape {tuple(emb.shape)}"
        self.output_dim = int(emb.shape[-1])

    def _prep(self, pixels: torch.Tensor) -> torch.Tensor:
        """Convert raw pixels to ImageNet-normalized floats."""
        if pixels.dtype == torch.uint8:
            pixels = pixels.float() / 255.0
        elif pixels.dtype != torch.float32:
            pixels = pixels.float()
        # (B, T, 3, H, W) — normalize per-channel; Normalize handles any
        # leading batch dims.
        return self._normalize(pixels)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode pixel frames to flat CLS embeddings.

        Uses ``torch.no_grad`` rather than ``inference_mode`` when
        frozen: the returned tensor must be usable as the **target** of
        a loss in the surrounding training loop, and inference_mode
        tensors cannot participate in autograd even as detached inputs.

        Args:
            pixels: (B, T, 3, H, W) — T can be 1 for single frames.
        Returns:
            emb: (B, T, D) CLS embedding per frame.
        """
        pixels = self._prep(pixels)
        if self.frozen:
            with torch.no_grad():
                info = self.jepa.encode({"pixels": pixels})
        else:
            info = self.jepa.encode({"pixels": pixels})
        return info["emb"]

    # nn.Module.forward — lets the adapter be used in torch.jit or
    # nn.Sequential contexts if ever needed.
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.encode(pixels)


def load_lewm_jepa_from_checkpoint(
    ckpt_path: str,
    lewm_repo_path: str = "external/le-wm",
    device: str = "cuda",
) -> nn.Module:
    """Rebuild a standalone LeWM ``jepa.JEPA`` instance from a Lightning
    checkpoint saved by LeWM's stable_pretraining Module.

    Strategy: rather than reconstructing the full Lightning Module
    (which needs stable_pretraining installed), we rebuild a plain
    JEPA and load the weights from the "model.*" prefix of the
    checkpoint's state_dict.

    Args:
        ckpt_path: path to a Lightning ``.ckpt`` (LeWM writes these
            as ``~/.stable_worldmodel/<run_name>_weights.ckpt``).
        lewm_repo_path: path to the LeWM source (for jepa.py imports).
        device: torch device to place the model on.
    """
    import sys
    from pathlib import Path

    lewm_src = Path(lewm_repo_path).resolve()
    if str(lewm_src) not in sys.path:
        sys.path.insert(0, str(lewm_src))

    # Delay-import so we can pre-insert the sys.path.
    import jepa  # type: ignore[import-not-found]
    from transformers import ViTConfig, ViTModel  # LeWM's encoder class

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]

    # LeWM stores hyper_parameters on the Lightning Module. We care
    # about the encoder shape + action encoder shape; read from the
    # checkpoint config if available, else fall back to ViT-tiny
    # defaults matching configs/train/lewm.yaml at img_size=224
    # patch_size=14 encoder_scale=tiny.
    def _vit_tiny_config() -> ViTConfig:
        return ViTConfig(
            hidden_size=192,
            num_hidden_layers=12,
            num_attention_heads=3,
            intermediate_size=768,
            image_size=224,
            patch_size=14,
        )

    # Build encoder
    enc_cfg = _vit_tiny_config()
    encoder = ViTModel(enc_cfg, add_pooling_layer=False)

    # Infer action encoder shape from checkpoint. LeWM's action encoder
    # is module.Embedder(input_dim=29, emb_dim=action_emb_dim). Check
    # the patch_embed weight shape to discover dims.
    ae_keys = [k for k in state if "action_encoder.patch_embed.weight" in k]
    if ae_keys:
        # Shape: (smoothed_dim, input_dim, 1)
        ae_weight = state[ae_keys[0]]
        action_input_dim = int(ae_weight.shape[1])
        action_smoothed_dim = int(ae_weight.shape[0])
    else:
        # Fall back to G1 defaults: 29-D action, 10-D smoothed.
        action_input_dim = 29
        action_smoothed_dim = 10

    # Discover emb_dim from the second Linear layer in action_encoder.
    emb_dim_keys = [k for k in state if "action_encoder.embed.2.weight" in k]
    if emb_dim_keys:
        action_emb_dim = int(state[emb_dim_keys[0]].shape[0])
    else:
        action_emb_dim = 10

    # Import Embedder + MLP from LeWM source.
    import module as lewm_module  # type: ignore[import-not-found]

    action_encoder = lewm_module.Embedder(
        input_dim=action_input_dim,
        smoothed_dim=action_smoothed_dim,
        emb_dim=action_emb_dim,
    )

    # LeWM's JEPA.encode() applies self.projector to the CLS token
    # before returning; we MUST reconstruct the projector to get the
    # same embedding space the trained model produced. Train-time
    # definition: MLP(in=hidden, hidden=2048, out=embed_dim,
    # norm=BatchNorm1d) => Linear(192,2048) -> BN -> GELU ->
    # Linear(2048,192). Matches configs/train/lewm.yaml default.
    projector = lewm_module.MLP(
        input_dim=enc_cfg.hidden_size,
        output_dim=enc_cfg.hidden_size,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )
    pred_proj = lewm_module.MLP(
        input_dim=enc_cfg.hidden_size,
        output_dim=enc_cfg.hidden_size,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    # Strip Lightning's "model." prefix once — reused for shape probes
    # and for load_state_dict.
    stripped = {
        k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")
    }

    # Predictor is LeWM's ARPredictor. Needed for L1 MPPI (two-level
    # hierarchical planning). Infer shape from checkpoint: num_frames
    # from pos_embedding, input/hidden dims from the pos_embedding's
    # trailing dim.
    pos_key = "predictor.pos_embedding"
    if pos_key in stripped:
        pos = stripped[pos_key]
        ar_num_frames = int(pos.shape[1])
        ar_input_dim = int(pos.shape[2])
    else:
        ar_num_frames, ar_input_dim = 3, enc_cfg.hidden_size
    # Per lewm.yaml defaults: depth=6, heads=16, mlp_dim=2048,
    # dim_head=64, dropout=0.1. These are fixed in the LeWM recipe;
    # the train.py builds ARPredictor with input_dim=hidden_dim=output_dim
    # = encoder.hidden_size.
    predictor = lewm_module.ARPredictor(
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

    # Rebuild proprio encoder if the checkpoint carries one (v3+).
    # Shape: MLP(proprio_dim -> hidden=256 -> embed_dim, norm=None)
    # matching train.py. Infer proprio_dim from net.0.weight, embed_dim
    # from net.3.weight.
    proprio_encoder = None
    prop_w0 = stripped.get("proprio_encoder.net.0.weight")
    prop_w3 = stripped.get("proprio_encoder.net.3.weight")
    if prop_w0 is not None and prop_w3 is not None:
        proprio_in = int(prop_w0.shape[1])
        proprio_hidden = int(prop_w0.shape[0])
        proprio_out = int(prop_w3.shape[0])
        proprio_encoder = lewm_module.MLP(
            input_dim=proprio_in,
            output_dim=proprio_out,
            hidden_dim=proprio_hidden,
            norm_fn=None,
        )

    jepa_model = jepa.JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
        proprio_encoder=proprio_encoder,
    )

    # Load all "model.*" state_dict entries. Predictor is now the real
    # ARPredictor, so we keep predictor.* keys too.
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
