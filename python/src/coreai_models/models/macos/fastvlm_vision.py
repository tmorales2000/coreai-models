"""FastVLM vision encoder + projector for Core AI export.

Standalone re-implementation of the FastVLM vision pipeline that loads
directly from the HF safetensors checkpoint.

Architecture::

  pixel_values [1, 3, 1024, 1024]  (fp32 input, cast to fp16 on entry)
    -> FastViTHD (re-authored — see note below)
       - forward_embeddings: patch_embed stem
       - forward_tokens: network stages (RepMixer + Attention)
       - conv_exp: expansion conv  [B, 3072, 16, 16]
    -> reshape + transpose        [B, 256, 3072]
    -> mlp2x_gelu projector
       - Linear(3072, hidden) + GELU + Linear(hidden, hidden)
  -> image_features [1, 256, hidden_size]

Variant specs (from HF config.json):

  Variant   hidden  mm_hidden  layers        max_ctx
  -------   ------  ---------  ------        -------
  0.5B      896     3072       24            32768
  1.5B      1536    3072       28            32768
  7B        3584    3072       28            32768

Image constants (all variants, from MobileCLIPVisionTower):

  IMAGE_SIZE   = 1024  (from config.mm_vision_tower = "mobileclip_l_1024")
  PATCH_SIZE   = 64    (conv_exp output: 1024/16 = 64, then /4 = 16 → actually stride)
  NUM_TOKENS   = 256   = (1024 // 64)^2  -- conv_exp 16x16 output, reshaped
  image_mean   = [0.0, 0.0, 0.0]        (CLIP normalization, no-op center)
  image_std    = [1.0, 1.0, 1.0]        (identity)
  image_strategy = "center_crop"        (CLIP-based preprocessor)

WHY RE-AUTHORING IS NEEDED (CoreML / ANE compiler)
---------------------------
FastVLM's FastViTHD encoder uses:
  1. LayerNormChannel — manual channel-wise layer norm over [B,C,H,W] that
     prevents ANE recognition as the layer_norm composite op.
     ANELayerNorm replaces it with F.layer_norm after a [B,C,H,W]→[B,H,W,C]
     permute, enabling ANE optimization.
  2. MHSA — manual QKV attention that prevents SDPA composite op recognition.
     ANEAttention replaces it using coreai_torch.composite_ops.SDPA.

Weights are already in inference mode in the checkpoint (all reparameterization
for MobileOneBlock, RepMixer, ReparamLargeKernelConv, RepCPE already applied
during FastVLM's training pipeline — each block is just Conv2d in the checkpoint).

WEIGHT KEY MAPPING
------------------
Vision tower (doubly nested)::

  model.vision_tower.vision_tower.model.* -> FastViT model.*

Projector::

  model.mm_projector.N.* -> layers.N.*

``from_pretrained`` handles these key mappings internally.

HF model IDs::

  fastvlm-0.5b: apple/FastVLM-0.5B
  fastvlm-1.5b: apple/FastVLM-1.5B
  fastvlm-7b:   apple/FastVLM-7B
"""

import glob
import os
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from coreai_torch.composite_ops import SDPA
from safetensors import safe_open

# ---------------------------------------------------------------------------
# ANE-compatible layer replacements
# ---------------------------------------------------------------------------


class ANELayerNorm(nn.Module):
    """Drop-in for LayerNormChannel using F.layer_norm.

    LayerNormChannel normalizes over channels on a [B,C,H,W] tensor manually
    (mean/pow/sqrt), preventing ANE recognition. This permutes to [B,H,W,C],
    applies F.layer_norm (recognized as the layer_norm composite op), then
    permutes back. Holds weight/bias as direct parameters matching the original
    key names so checkpoint loading requires no key renaming.
    """

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, [x.shape[-1]], self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class ANEAttention(nn.Module):
    """Drop-in for FastViT MHSA using coreai_torch.composite_ops.SDPA.

    Original MHSA computes attention as q@k.T * scale → softmax → @v, which
    prevents ANE compiler recognition. SDPA composite op replaces this.
    Input/output: [B, C, H, W] — flattened to [B, N, C] for attention.
    """

    def __init__(self, dim: int, head_dim: int = 32) -> None:
        super().__init__()
        assert dim % head_dim == 0
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.sdpa = SDPA(is_causal=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.flatten(2).transpose(1, 2)  # [B, N, C]
        qkv = (
            self.qkv(x_flat).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # each [B, num_heads, N, head_dim]
        out = self.sdpa(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------


def _patch_layernorm(model: nn.Module, LayerNormChannel) -> None:
    """Replace all LayerNormChannel instances with ANELayerNorm in-place."""
    for name, module in model.named_modules():
        if isinstance(module, LayerNormChannel):
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            replacement = ANELayerNorm(num_features=module.weight.shape[0], eps=module.eps)
            replacement.weight = nn.Parameter(module.weight.clone())
            replacement.bias = nn.Parameter(module.bias.clone())
            setattr(parent, parts[-1], replacement)


def _patch_attention(model: nn.Module, MHSA) -> None:
    """Replace all MHSA instances with ANEAttention in-place."""
    for name, module in model.named_modules():
        if isinstance(module, MHSA):
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            dim = module.qkv.in_features
            replacement = ANEAttention(dim=dim, head_dim=module.head_dim)
            replacement.qkv.weight = nn.Parameter(module.qkv.weight.clone())
            replacement.proj.weight = nn.Parameter(module.proj.weight.clone())
            replacement.proj.bias = nn.Parameter(module.proj.bias.clone())
            setattr(parent, parts[-1], replacement)


# ---------------------------------------------------------------------------
# FastVLM vision model (FastViTHD + mlp2x_gelu projector)
# ---------------------------------------------------------------------------

# Doubly-nested prefix confirmed from FastVLM checkpoint inspection.
_VISION_PREFIX = "model.vision_tower.vision_tower.model."
_PROJECTOR_PREFIX = "model.mm_projector."


class FastVLMVisionModel(nn.Module):
    """FastVLM vision encoder + projector for Core AI export.

    Combines FastViTHD (re-authored for ANE) with the mlp2x_gelu projector
    into a single module, matching the CastF16VisionEncoder contract:

      Input:  pixel_values float32 [1, 3, 1024, 1024]
      Output: image_features float16 [1, 256, hidden_size]

    The input is fp32 per the Swift runner contract. The internal cast to fp16
    happens as the first op inside forward(), so the compiled aimodel accepts
    fp32 pixel values while running fp16 internally.

    FastViT is instantiated directly from llava_qwen.py (loaded from the
    weights_dir) rather than via timm.create_model("fastvithd"), which returns
    an empty model in this environment. The exact config comes from the
    fastvithd() factory function in llava_qwen.py (lines 1493-1519).
    """

    NUM_IMAGE_TOKENS = 256  # (1024 // 64)^2 — conv_exp 16x16 output

    def __init__(self, weights_dir: str, config) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        mm_hidden = getattr(config, "mm_hidden_size", 3072)
        proj_type = getattr(config, "mm_projector_type", "mlp2x_gelu")

        # ---- Vision tower (FastViTHD) ----
        sys.path.insert(0, weights_dir)
        try:
            import llava_qwen
        finally:
            sys.path.pop(0)

        self.vision_tower = llava_qwen.FastViT(
            layers=[2, 12, 24, 4, 2],
            embed_dims=[96, 192, 384, 768, 1536],
            mlp_ratios=[4, 4, 4, 4, 4],
            downsamples=[True, True, True, True, True],
            token_mixers=["repmixer", "repmixer", "repmixer", "attention", "attention"],
            norm_layer=llava_qwen.LayerNormChannel,
            pos_embs=[
                None,
                None,
                None,
                llava_qwen.RepCPE,  # class, not instance — FastViT.__init__ instantiates
                llava_qwen.RepCPE,
            ],
            inference_mode=True,
        )
        _patch_layernorm(self.vision_tower, llava_qwen.LayerNormChannel)
        _patch_attention(self.vision_tower, llava_qwen.MHSA)

        # ---- Projector (mlp2x_gelu) ----
        match = re.match(r"^mlp(\d+)x_gelu$", proj_type)
        depth = int(match.group(1)) if match else 2
        layers: list[nn.Module] = [nn.Linear(mm_hidden, hidden_size)]
        for _ in range(1, depth):
            layers += [nn.GELU(), nn.Linear(hidden_size, hidden_size)]
        self.projector = nn.Sequential(*layers)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: float32 [1, 3, 1024, 1024]  (Swift runner contract)

        Returns:
            image_features: float16 [1, 256, hidden_size]
        """
        # Cast to fp16 as first op; runs fp16 internally.
        x = pixel_values.to(dtype=next(self.parameters()).dtype)

        # FastViTHD forward — matches MobileCLIPVisionTower.feature_select()
        x = self.vision_tower.forward_embeddings(x)  # patch_embed stem
        x = self.vision_tower.forward_tokens(x)  # network stages
        x = self.vision_tower.conv_exp(x)  # [B, 3072, 16, 16]

        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(1, 2)  # [B, 256, 3072]

        # Project to LM embedding space
        x = self.projector(x)  # [B, 256, hidden_size]
        return x

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        config,
        dtype: torch.dtype = torch.float32,
    ) -> "FastVLMVisionModel":
        """Load FastVLMVisionModel weights from the HF safetensors checkpoint.

        Loads the FastViTHD vision tower and mlp2x_gelu projector weights
        from the model's safetensors files, applying the key mapping:

          Vision:    model.vision_tower.vision_tower.model.* -> vision_tower.*
          Projector: model.mm_projector.N.*                  -> projector.N.*
        """
        model = cls(model_dir, config).to(dtype=dtype)

        st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        vision_weights: dict[str, torch.Tensor] = {}
        proj_weights: dict[str, torch.Tensor] = {}

        for path in st_files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118
                    t = f.get_tensor(key)
                    if t.dtype != dtype:
                        t = t.to(dtype)
                    if key.startswith(_VISION_PREFIX):
                        vision_weights[key.removeprefix(_VISION_PREFIX)] = t
                    elif key.startswith(_PROJECTOR_PREFIX):
                        # "model.mm_projector.0.weight" → "0.weight"
                        proj_weights[key.removeprefix(_PROJECTOR_PREFIX)] = t

        # Load vision tower (head.* and .bn.* keys are inference-mode artifacts,
        # not used in forward(); exclude from unexpected-key check.)
        missing, unexpected = model.vision_tower.load_state_dict(
            vision_weights, assign=True, strict=False
        )
        unexpected_real = [
            k
            for k in unexpected
            if "head" not in k and ".bn." not in k and "num_batches_tracked" not in k
        ]
        missing_real = [k for k in missing if "head" not in k]
        if unexpected_real:
            raise RuntimeError(f"FastVLM vision tower unexpected keys: {unexpected_real[:5]}")
        if missing_real:
            raise RuntimeError(f"FastVLM vision tower missing keys: {missing_real[:5]}")

        # Load projector (strict — all keys must match)
        model.projector.load_state_dict(proj_weights, assign=True, strict=True)

        return model
