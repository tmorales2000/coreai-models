"""FastVLM text decoder (Qwen2-based) for Core AI VLM export.

FastVLM uses the Qwen2 language model with:
  - attention_bias = True (QKV have bias)
  - tie_word_embeddings = False (0.5B, 1.5B, 7B)
  - rope_theta = 1000000.0

This file provides the ``inputs_embeds`` variant used by the VLM export path.
The token embedding lookup is exported separately as ``embed.aimodel``, so the
decoder takes pre-computed embeddings rather than token IDs.

Text decoder architecture (all three variants share the same topology):

  inputs_embeds [1, L, hidden]
    -> 28 (or 24) TransformerBlocks (Qwen2 attention + SwiGLU MLP)
    -> RMSNorm
    -> lm_head Linear
  -> logits [1, L, vocab_size]

Key differences from the LLM export path (Qwen2ForCausalLM):
  - forward() takes inputs_embeds, not input_ids (embed_tokens is excluded)
  - Weights loaded from model.language_model.* prefix (VL checkpoint layout)
  - embed_tokens weights are dropped (live in embed.aimodel)
  - No prefill_graph export (VLM path uses a single graph)
"""

from typing import Any

import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM as HFQwen2ForCausalLM
from typing_extensions import Self, override

from coreai_models._hf import resolve_rope_theta
from coreai_models.models.base import BaseForCausalLM
from coreai_models.models.macos.qwen2 import TransformerBlock
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.rms_norm import RMSNorm

_DEFAULT_ROPE_THETA = 1_000_000


class FastVLMModelEmbeddings(nn.Module):
    """Qwen2 transformer stack that takes pre-computed embeddings (no embed_tokens)."""

    def __init__(self, config) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        h = inputs_embeds
        for layer in self.layers:
            h = layer(h, position_ids, cache)
        return self.norm(h)


class FastVLMForCausalLMEmbeddings(BaseForCausalLM):
    """FastVLM text decoder for Core AI VLM export (inputs_embeds variant).

    forward(inputs_embeds, position_ids, k_cache, v_cache) -> logits

    Takes pre-computed token + image embeddings as inputs_embeds.
    The token embedding lookup lives in the separate embed.aimodel component.

    Loaded from the VL checkpoint key prefix ``model.language_model.*``
    (same layout as Qwen3-VL). embed_tokens weights are dropped; all
    remaining text-decoder weights are fused and loaded.
    """

    _HF_MODEL_CLASS = HFQwen2ForCausalLM

    @classmethod
    def _get_reauthored_config(
        cls,
        hf_config,
        max_context_length: int | None = None,
        num_layers: int | None = None,
    ):
        # FastVLM HF config nests text config under text_config or llm_config.
        text_config = (
            getattr(hf_config, "text_config", None)
            or getattr(hf_config, "llm_config", None)
            or hf_config
        )
        if max_context_length is not None:
            text_config.max_position_embeddings = max_context_length
        if num_layers is not None:
            text_config.num_hidden_layers = num_layers
        text_config.tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)
        rope_theta = resolve_rope_theta(text_config, default=_DEFAULT_ROPE_THETA)
        text_config.rope_theta = float(rope_theta)
        text_config.rope_scaling = None
        return text_config

    @override
    def _init_model(self, config) -> None:
        self.model = FastVLMModelEmbeddings(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        cache = KVCache(k_cache, v_cache)
        out = self.model(inputs_embeds, position_ids, cache)
        return self.lm_head(out)

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        # Strip "model.language_model." prefix from VL checkpoint keys.
        for key in list(state_dict.keys()):
            if key.startswith("model.language_model."):
                new_key = "model." + key[len("model.language_model.") :]
                state_dict[new_key] = state_dict.pop(key)
            elif key.startswith("model.vision_tower.") or key.startswith("model.mm_projector."):
                del state_dict[key]

        # Drop embed_tokens — lives in embed.aimodel.
        for k in list(state_dict.keys()):
            if "embed_tokens" in k:
                del state_dict[k]

        # Fuse q_proj, k_proj, v_proj -> qkv_proj (matching Qwen2ForCausalLM._mutate_state_dict).
        max_layer = -1
        for k in state_dict:
            if k.startswith("model.layers."):
                parts = k.split(".")
                if len(parts) >= 3:
                    max_layer = max(max_layer, int(parts[2]))

        if max_layer < 0:
            raise ValueError("invalid state_dict: no transformer layers found")

        expects_bias = getattr(self.config, "attention_bias", True)

        for i in range(max_layer + 1):
            combined_weight = []
            combined_bias = []
            has_weights = True
            has_bias = True
            for proj in ["q_proj", "k_proj", "v_proj"]:
                wk = f"model.layers.{i}.self_attn.{proj}.weight"
                bk = f"model.layers.{i}.self_attn.{proj}.bias"
                if wk not in state_dict:
                    has_weights = False
                    continue
                combined_weight.append(state_dict.pop(wk))
                if bk in state_dict:
                    if expects_bias:
                        combined_bias.append(state_dict[bk])
                    del state_dict[bk]
                else:
                    has_bias = False
            if has_weights and combined_weight:
                state_dict[f"model.layers.{i}.self_attn.qkv_proj.weight"] = torch.concat(
                    combined_weight, axis=0
                )
                if expects_bias and has_bias and combined_bias:
                    state_dict[f"model.layers.{i}.self_attn.qkv_proj.bias"] = torch.concat(
                        combined_bias, axis=0
                    )

    @classmethod
    @override
    def export_input_names(cls) -> dict[str, tuple[str, ...]]:
        """Override: FastVLM decoder takes ``inputs_embeds``, not ``input_ids``.

        The token embedding lookup lives in the separate embed.aimodel component,
        so the decoder receives pre-computed embeddings.  The runner-visible input
        name in the compiled bundle is ``inputs_embeds`` (not the base-class default
        ``input_ids``).
        """
        from coreai_models._constants import MAIN_GRAPH_NAME

        return {MAIN_GRAPH_NAME: ("inputs_embeds", "position_ids")}

    @override
    def build_reference_inputs(
        self, config, target_dtype, spec
    ) -> dict[str, dict[str, torch.Tensor]]:
        from coreai_models._constants import MAIN_GRAPH_NAME

        hidden = config.hidden_size
        n_layers = config.num_hidden_layers
        n_kv = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", hidden // config.num_attention_heads)
        max_ctx = getattr(spec, "cache_seq_len", config.max_position_embeddings)
        qlen = getattr(spec, "query_len", 64)
        offset = getattr(spec, "offset", 64)

        return {
            MAIN_GRAPH_NAME: {
                "inputs_embeds": torch.randn(1, qlen, hidden, dtype=target_dtype),
                "position_ids": torch.arange(qlen + offset, dtype=torch.int32).unsqueeze(0),
                "k_cache": torch.zeros(n_layers, 1, n_kv, max_ctx, head_dim, dtype=target_dtype),
                "v_cache": torch.zeros(n_layers, 1, n_kv, max_ctx, head_dim, dtype=target_dtype),
            }
        }

    @override
    def build_dynamic_shapes(self, config, spec) -> dict[str, dict[str, Any]]:
        from coreai_models._constants import MAIN_GRAPH_NAME

        max_ctx = getattr(spec, "cache_seq_len", config.max_position_embeddings)
        qlen = getattr(spec, "query_len", 64)

        return {
            MAIN_GRAPH_NAME: {
                "inputs_embeds": {1: torch.export.Dim("query_len", max=max_ctx - 2)},
                "position_ids": {1: torch.export.Dim("seq_pos", min=qlen, max=max_ctx - 1)},
                "k_cache": None,
                "v_cache": None,
            }
        }
