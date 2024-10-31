
import functools
import math
from types import MethodType
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaModel
from transformers.cache_utils import Cache, StaticCache
from transformers.models.llama.modeling_llama import (LlamaAttention,
                                                      _flash_attention_forward,
                                                      apply_rotary_pos_emb,
                                                      repeat_kv)
from transformers.utils import logging

from speculative_prefill.vllm_patch.config import SpecConfig

logger = logging.get_logger(__name__)


def _custom_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    spec_config: Optional[SpecConfig] = None
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
        assert position_ids is not None
        assert attention_mask is None
        """
            pos_ids: [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 0, 0]
            last_token_mask: 
                - [T, F, F, F, T, F, F, F, T, F, T, T]
                - [F, F, F, T, F, F, F, T, F, T, T, T]
        """
        last_token_mask = (position_ids == 0).view(-1)
        last_token_mask = torch.nn.functional.pad(
            last_token_mask[1:], (0, 1), value=1.0).to(torch.bool)
        seq_lens = torch.nonzero((last_token_mask == True)).view(-1)
        seq_lens = torch.cat([seq_lens[:1] + 1, seq_lens[1:] - seq_lens[:-1]], dim=-1)
        
        # size will be [1, head, seqlen, head_dim]
        keys = repeat_kv(key_states, self.num_key_value_groups)
        keys = torch.split(keys, seq_lens.tolist(), dim=-2)
        # size will be [1, head, seqlen, head_dim]
        querys = query_states[:, :, last_token_mask, :]

        # compute attn weights per sample
        attn_weights = []
        for sample_idx, key in enumerate(keys):
            query = querys[:, :, sample_idx, :].unsqueeze(-2)
            attn = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn = nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(querys.dtype)
            attn_weights.append(attn)
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


def enable_fa2_output_attns(
    model: LlamaModel, 
    spec_config: SpecConfig
) -> LlamaModel:
    for layer_idx in range(model.config.num_hidden_layers):
        self_attn: LlamaAttention = model.layers[layer_idx].self_attn
        self_attn.forward = MethodType(
            functools.partial(
                _custom_forward, 
                spec_config=spec_config
            ), self_attn
        )
    return model
