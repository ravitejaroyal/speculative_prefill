"""
    A prefill speculator will:
    1. first get the full context, predict the thrown away indices based on the
        first layer
    2. send the indices to the main model
    3. finish the full prefill and send the KV cache to the main model

    A new prefill speculator will:
    1. get the full context, generate N tokens. 
    2. these N tokens are used to access what tokens are useful for the context
    3. send the indices to the main model
"""

import atexit
import json
import math
import os
from dataclasses import dataclass
from types import MethodType
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers.cache_utils import Cache, StaticCache
from transformers.models.llama.modeling_llama import (_flash_attention_forward,
                                                      apply_rotary_pos_emb,
                                                      repeat_kv)
from transformers.utils import logging

from models import SAVE_STATS_DIR

logger = logging.get_logger(__name__)


@dataclass
class SpeculativePrefillData:
    keep_indices: Optional[torch.Tensor] = None


class SpeculativePrefillStats:
    def __init__(self):
        self.num_of_queries = 0
        self.dropped_token_cnts = []
        self.keep_ratios = []
        self.save_stats_dir = SAVE_STATS_DIR
        if self.save_stats_dir is not None:
            atexit.register(self.save_results)
    
    def update(self, dropped_token_cnt, keep_ratio):
        self.num_of_queries += 1
        self.dropped_token_cnts.append(dropped_token_cnt)
        self.keep_ratios.append(keep_ratio)

    def save_results(self):
        print(f"Saving stats to {self.save_stats_dir}...")
        os.makedirs(self.save_stats_dir, exist_ok=True)
        # save basic info
        with open(os.path.join(self.save_stats_dir, "stats.json"), 'w') as f:
            json.dump({
                "num_of_queries": self.num_of_queries, 
                "dropped_token_cnts": self.dropped_token_cnts, 
                "keep_ratios": self.keep_ratios, 
                "avg_ratio": sum(self.keep_ratios) / self.num_of_queries if self.num_of_queries > 0 else -1
            }, f)
        
        # figure on the distribution of percentage of drops
        sns.histplot(
            self.keep_ratios, 
            bins=20, 
            kde=True, 
            color='skyblue', 
            edgecolor='black'
        )

        plt.title('Distribution of Speculative Prefill Keep Ratios', fontsize=16)
        plt.xlabel('Prefill Token Keep Ratio', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)

        plt.savefig(os.path.join(self.save_stats_dir, "figure.png"), dpi=500)


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
        monkey patch on outputing the desired attention behaviors
    """
    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
            "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
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

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    if output_attentions:
        # we dont need all but just the last for each forward
        # we just need [B, H, Q, K] and Q can be 1
        # [B, head, seqlen, head_dim]
        keys = repeat_kv(key_states, self.num_key_value_groups)
        # [B, head, 1, head_dim]
        querys = query_states[..., -1:, :]
        attn_weights = torch.matmul(querys, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(querys.dtype)
    else:
        attn_weights = None

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
        sliding_window=getattr(self, "sliding_window", None),
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        is_causal=self.is_causal,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights, past_key_value


def build_speculator(device: Optional[torch.device] = None) -> LlamaForCausalLM:
    speculator: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B-Instruct', 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        device_map=device, 
        attn_implementation="flash_attention_2", # eager is required to output attn scores
        trust_remote_code=True
    )

    speculator.spec_prefill_stats = SpeculativePrefillStats()

    for layer_idx in range(speculator.config.num_hidden_layers):
        self_attn = speculator.model.layers[layer_idx].self_attn
        self_attn.forward = MethodType(
            forward, self_attn
        )

    # saves around 10GB mem
    for name, param in speculator.named_parameters():
        if name != "model.embed_tokens.weight":
            param.requires_grad_(False)

    return speculator


def spec_prefill_data_to_inputs(
    spec_prefill_data: SpeculativePrefillData, 
    input_ids: torch.LongTensor, 
    attention_mask: Optional[torch.Tensor] = None, 
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
        'attention_mask': attention_mask, 
        'original_seq_len': S
    }


from models.speculator.algorithms import (speculate_tokens_based_on_attn,
                                          speculate_tokens_based_on_grad)

SPECULATOR_ALGO = {
    "attn": speculate_tokens_based_on_attn, 
    "grad": speculate_tokens_based_on_grad
}