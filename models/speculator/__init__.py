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
from typing import Dict, Optional

import torch
from transformers import (AutoModelForCausalLM, GenerationConfig,
                          LlamaForCausalLM)


@dataclass
class SpeculativePrefillData:
    keep_indices: Optional[torch.Tensor] = None


def build_speculator(device: Optional[torch.device] = None) -> LlamaForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B-Instruct', 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        device_map=device, 
        attn_implementation="eager", # eager is required to output attn scores
        trust_remote_code=True
    )


def speculate_tokens(
    speculator: LlamaForCausalLM, 
    input_ids: torch.Tensor, 
    attention_mask: Optional[torch.Tensor] = None, 
    decode_cnt: int = 8, 
    keep: float = -1, 
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
        max_new_tokens=decode_cnt, 
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

    if keep == -1:
        topk = prefill_len
    elif 0.0 < keep < 1.0:
        topk = math.ceil(prefill_len * keep)
    else:
        topk = min(prefill_len, keep)

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