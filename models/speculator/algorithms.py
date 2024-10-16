import math
from typing import Optional

import torch
from transformers import GenerationConfig, LlamaForCausalLM
from transformers.utils import logging

from models import VERBOSITY
from models.speculator import SpeculativePrefillData

logger = logging.get_logger(__name__)


def speculate_tokens_based_on_attn(
    speculator: LlamaForCausalLM, 
    input_ids: torch.Tensor, 
    attention_mask: Optional[torch.Tensor] = None, 
    look_ahead_cnt: int = 8, 
    keep: float = -2, 
    gen_config: Optional[GenerationConfig] = None, 
    **kwargs
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

    if keep == -3:
        # adaptive strategy based on max * ratio threshold
        max_quant_attns = torch.quantile(all_attns.float(), q=0.99, dim=-1, keepdim=True)
        threshold = max_quant_attns * 0.005
        # sum to get number of tokens, max over batch
        topk = (all_attns > threshold).sum(-1).max(0)[0].item()
    elif keep == -2:
        # adaptive strategy based on max * ratio threshold
        max_attns = torch.max(all_attns, dim=-1, keepdim=True)[0]
        threshold = max_attns * 0.001
        # sum to get number of tokens, max over batch
        topk = (all_attns > threshold).sum(-1).max(0)[0].item()
    elif keep == -1:
        # keep all strategy
        topk = prefill_len
    elif 0.0 < keep < 1.0:
        # fixed percentage strategy
        topk = math.ceil(prefill_len * keep)
    else:
        # absolute count strategy
        topk = min(prefill_len, keep)

    if VERBOSITY >= 1:
        print(f"Keep strategy = {keep}, Kept token percentage = {(topk / prefill_len) * 100:.2f}%, Look ahead cnt = {look_ahead_cnt}.")

    _, keep_indices = torch.topk(all_attns, dim=-1, k=topk)

    speculator.spec_prefill_stats.update(
        dropped_token_cnt=prefill_len - topk, 
        keep_ratio=(topk / prefill_len)
    )

    return SpeculativePrefillData(keep_indices=keep_indices)


def speculate_tokens_based_on_grad(
    speculator: LlamaForCausalLM, 
    input_ids: torch.Tensor, 
    attention_mask: Optional[torch.Tensor] = None, 
    look_ahead_cnt: int = 8, 
    keep: float = -2, 
    gen_config: Optional[GenerationConfig] = None, 
    **kwargs
) -> SpeculativePrefillData:
    
    prefill_len = input_ids.shape[-1]

    with torch.enable_grad():
        if not speculator.model.gradient_checkpointing:
            speculator.model.gradient_checkpointing_enable()
        # set required for using gradient checkpointing
        speculator.model.training = True

        inputs_embeds: torch.Tensor = speculator.model.embed_tokens(input_ids)
        inputs_embeds.retain_grad()

        outputs = speculator.forward(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask, 
            use_cache=False,  
            return_dict=True
        )

        # set it back
        speculator.model.training = False

        # [B, seqlen, logits]
        logits = outputs.logits
        
        # [B, logits]
        decode_token = logits[:, -1, :]

        # noisy target
        noise = torch.randn_like(decode_token)
        noise = noise / torch.linalg.vector_norm(noise, dim=-1)
        target = decode_token.detach() + noise

        # compute the loss (CE seems to be much better than MSE)
        loss = torch.nn.functional.cross_entropy(decode_token, target)
        loss.backward()

        # get the gradients
        grads = inputs_embeds.grad
        grad_magnitudes = torch.linalg.vector_norm(grads, dim=-1)

    if keep == -3:
        # adaptive strategy based on max * ratio threshold
        max_quant_attns = torch.quantile(grad_magnitudes.float(), q=0.99, dim=-1, keepdim=True)
        threshold = max_quant_attns * 0.01
        # sum to get number of tokens, max over batch
        topk = (grad_magnitudes > threshold).sum(-1).max(0)[0].item()
    elif keep == -2:
        # adaptive strategy based on max * ratio threshold
        max_attns = torch.max(grad_magnitudes, dim=-1, keepdim=True)[0]
        threshold = max_attns * 0.01
        # sum to get number of tokens, max over batch
        topk = (grad_magnitudes > threshold).sum(-1).max(0)[0].item()
    elif keep == -1:
        # keep all strategy
        topk = prefill_len
    elif 0.0 < keep < 1.0:
        # fixed percentage strategy
        topk = math.ceil(prefill_len * keep)
    else:
        # absolute count strategy
        topk = min(prefill_len, keep)

    if VERBOSITY >= 1:
        print(f"Keep strategy = {keep}, Kept token percentage = {(topk / prefill_len) * 100:.2f}%, Look ahead cnt = {look_ahead_cnt}.")

    _, keep_indices = torch.topk(grad_magnitudes, dim=-1, k=topk)

    speculator.spec_prefill_stats.update(
        dropped_token_cnt=prefill_len - topk, 
        keep_ratio=(topk / prefill_len)
    )

    return SpeculativePrefillData(keep_indices=keep_indices)