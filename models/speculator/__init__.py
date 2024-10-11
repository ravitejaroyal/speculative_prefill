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

import math
from dataclasses import dataclass
from types import MethodType
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (AutoModelForCausalLM, GenerationConfig,
                          LlamaForCausalLM)
from transformers.cache_utils import Cache, StaticCache
from transformers.models.llama.modeling_llama import (_flash_attention_forward,
                                                      apply_rotary_pos_emb,
                                                      repeat_kv)
from transformers.utils import logging

from models import VERBOSITY

logger = logging.get_logger(__name__)


@dataclass
class SpeculativePrefillData:
    keep_indices: Optional[torch.Tensor] = None


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

    for layer_idx in range(speculator.config.num_hidden_layers):
        self_attn = speculator.model.layers[layer_idx].self_attn
        self_attn.forward = MethodType(
            forward, self_attn
        )

    return speculator


def speculate_tokens(
    speculator: LlamaForCausalLM, 
    input_ids: torch.Tensor, 
    attention_mask: Optional[torch.Tensor] = None, 
    look_ahead_cnt: int = 8, 
    keep: float = -2, 
    gen_config: Optional[GenerationConfig] = None
) -> SpeculativePrefillData:

    if gen_config is None:
        gen_config = GenerationConfig(
            do_sample=False, 
            eos_token_id=128009, 
            pad_token_id=128009
        )

    outputs = speculator.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,  
        max_new_tokens=look_ahead_cnt, 
        use_cache=True, 
        return_dict_in_generate=True, 
        output_attentions=True, 
        generation_config=gen_config
    )

    # dimension: [decoding_cnt, layer_cnt, [B, H, Q, K]]
    attentions = outputs.attentions
    prefill_len = input_ids.shape[-1]

    all_attns = []

    for pos in range(len(attentions)):
        # Tuple[B, H, Q, K]
        attn = attentions[pos]
        # [layer_cnt, B, H, Q, K]
        attn = torch.stack(attn, dim=0)
        # [layer_cn, B, 1, prefill_len]
        attn = attn.max(dim=2)[0][..., -1:, :prefill_len]
        # [B, prefill_len]
        attn = attn.mean(dim=0).squeeze(1)
        # aggregate
        all_attns.append(attn)
    
    # [B, prefill_len]
    all_attns = torch.max(torch.stack(all_attns, dim=0), dim=0)[0]

    # smoothing
    # all_attns = torch.avg_pool1d(all_attns, kernel_size=5, padding=5 // 2, stride=1)

    if keep == -3:
        # adaptive strategy based on quantile * ratio
        # this will remove the influence of the max outliers (e.g. sink tokens)
        quantile = torch.quantile(all_attns.float(), q=0.98, dim=-1, keepdim=True)
        threshold = quantile * 0.005
        # sum to get number of tokens, max over batch
        topk = (all_attns > threshold).sum(-1).max(0)[0]
    elif keep == -2:
        # adaptive strategy based on max * ratio threshold
        max_attns = torch.max(all_attns, dim=-1, keepdim=True)[0]
        threshold = max_attns * 0.001
        # sum to get number of tokens, max over batch
        topk = (all_attns > threshold).sum(-1).max(0)[0]
    elif keep == -1:
        # keep all strategy
        topk = prefill_len
    elif 0.0 < keep < 1.0:
        # fixed percentage strategy
        topk = math.ceil(prefill_len * keep)
    else:
        # absolute count strategy
        topk = min(prefill_len, keep)

    if VERBOSITY == 1:
        print(f"Keep strategy = {keep}, Kept token percentage = {(topk / prefill_len) * 100:.2f}%, Look ahead cnt = {look_ahead_cnt}.")

    _, keep_indices = torch.topk(all_attns, dim=-1, k=topk)

    return SpeculativePrefillData(keep_indices=keep_indices)


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