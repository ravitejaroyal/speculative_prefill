import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (LLAMA_ATTENTION_CLASSES,
                                                      LlamaAttention,
                                                      LlamaDecoderLayer,
                                                      LlamaModel, LlamaRMSNorm,
                                                      LlamaRotaryEmbedding,
                                                      apply_rotary_pos_emb,
                                                      repeat_kv)
from transformers.utils import logging

from .configuration_llama_spec_prefill import LlamaSpecPrefillConfig

logger = logging.get_logger(__name__)


@dataclass
class SpeculativePrefillData:
    keep_indices: Optional[torch.Tensor] = None


class LlamaSpecPrefillAttention(LlamaAttention):

    def __init__(self, config: LlamaSpecPrefillConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        
    def speculative_prefill(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> SpeculativePrefillData:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
            
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        
        # we take the last token's attn for prediction
        attn_weights = attn_weights[..., -1, :]
        seq_len = attn_weights.shape[-1]

        # max over heads
        keep_token_cnt = min(seq_len, self.config.keep_token_cnt)
        attn_weights_max = torch.max(attn_weights, dim=1, keepdim=True)[0]
        _, keep_indices = torch.topk(attn_weights_max, dim=-1, k=keep_token_cnt)

        return SpeculativePrefillData(
            keep_indices=keep_indices, 
        )


class LlamaSpecPrefillDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self, 
        config: LlamaSpecPrefillConfig, 
        layer_idx: int
    ):
        super().__init__(config, layer_idx)
        if layer_idx == 0:
            self.self_attn = LlamaSpecPrefillAttention(config=config, layer_idx=layer_idx)
        else:
            self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

    def speculative_prefill(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> SpeculativePrefillData:
        
        hidden_states = self.input_layernorm(hidden_states)
        return self.self_attn.speculative_prefill(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )


class LlamaSpecPrefillModel(LlamaModel):
    config_class = LlamaSpecPrefillConfig
    _no_split_modules = ["LlamaSpecPrefillDecoderLayer"]

    def __init__(self, config: LlamaSpecPrefillConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaSpecPrefillDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def speculative_prefill(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None, 
        **kwargs
    ) -> SpeculativePrefillData:
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, None, False
        )

        first_layer: LlamaSpecPrefillDecoderLayer = self.layers[0]
        return first_layer.speculative_prefill(
            hidden_states=hidden_states, 
            attention_mask=causal_mask, 
            position_ids=position_ids, 
            position_embeddings=position_embeddings, 
            **kwargs
        )

    def speculative_prefill_data_to_inputs(
        self, 
        spec_prefill_data: SpeculativePrefillData, 
        input_ids: torch.LongTensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B, S = input_ids.shape
        device = input_ids.device
        keep_indices = spec_prefill_data.keep_indices.view(B, -1).to(device)
        keep_indices = torch.sort(keep_indices, dim=-1)[0]

        # [B, S]
        position_ids = torch.arange(S, device=device).expand_as(input_ids).to(device).gather(-1, keep_indices)
        input_ids = input_ids.gather(-1, keep_indices)

        if attention_mask is not None:
            attention_mask = attention_mask.to(device).gather(-1, keep_indices)
        
        return {
            'input_ids': input_ids, 
            'position_ids': position_ids, 
            'attention_mask': attention_mask
        }
