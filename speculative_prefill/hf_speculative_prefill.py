import torch
from torch import nn
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


class HFFSpeculativePrefill:
    """Standalone speculative prefill implementation using HuggingFace models."""

    def __init__(self, base_model: str, spec_model: str, look_ahead: int = 4, keep_percentage: float = 0.5, device: str = "cuda"):
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model).to(device)
        self.spec_model = AutoModelForCausalLM.from_pretrained(spec_model).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.spec_tokenizer = AutoTokenizer.from_pretrained(spec_model)
        self.look_ahead = look_ahead
        self.keep_percentage = keep_percentage
        self.device = device
        self.base_model.eval()
        self.spec_model.eval()

    @torch.no_grad()
    def __call__(self, prompt: str) -> torch.Tensor:
        base_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        spec_ids = self.spec_tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        # 1. run base model to collect key cache from context
        base_out = self.base_model(base_ids, use_cache=True, output_hidden_states=True, return_dict=True)
        base_keys = [k.squeeze(0) for k, _ in base_out.past_key_values]
        base_hidden = base_out.hidden_states

        seq_len = base_ids.size(1)

        # 2. look-ahead speculation using speculator model
        spec_cache = None
        queries_per_step: List[torch.Tensor] = []
        new_tokens: List[torch.Tensor] = []
        for step in range(self.look_ahead):
            outputs = self.spec_model(spec_ids if spec_cache is None else spec_ids[:, -1:],
                                       use_cache=True, past_key_values=spec_cache,
                                       output_hidden_states=True, return_dict=True)
            spec_cache = outputs.past_key_values
            last_h = [h[:, -1:, :] for h in outputs.hidden_states[1:]]
            layer_queries = []
            for layer_idx, h in enumerate(last_h):
                attn = self.spec_model.model.layers[layer_idx].self_attn
                pos = torch.tensor([[seq_len + step]], device=self.device)
                cos, sin = self.spec_model.model.rotary_emb(h, pos)
                q = attn.q_proj(h)
                k = attn.k_proj(h)
                q = q.view(1, 1, attn.num_attention_heads, attn.head_dim).transpose(1, 2)
                k = k.view(1, 1, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
                q, _ = apply_rotary_pos_emb(q, k, cos, sin)
                layer_queries.append(q.squeeze(1))
            queries_per_step.append(torch.stack(layer_queries))
            next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
            new_tokens.append(next_token)
            spec_ids = torch.cat([spec_ids, next_token], dim=1)
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        queries = torch.stack(queries_per_step, dim=2)  # [layer, head, step, dim]
        keys = torch.stack(base_keys)  # [layer, head, seq, dim]
        attn = torch.einsum("lhsd, lhnd -> lhsn", queries, keys) / (queries.shape[-1] ** 0.5)
        attn = attn.mean(2).max(1)[0]
        token_importance = attn.mean(0)

        topk = max(1, int(seq_len * self.keep_percentage))
        keep_idx = torch.topk(token_importance, k=topk).indices.sort().values
        kept_ids = base_ids[:, keep_idx]

        final_ids = torch.cat([kept_ids] + new_tokens, dim=1)
        return self.base_model(final_ids).logits
