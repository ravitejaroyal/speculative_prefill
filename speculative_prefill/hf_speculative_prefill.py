import torch
from torch import nn
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


class HFFSpeculativePrefill:
    """Standalone speculative prefill implementation using HuggingFace models.

    This follows the same logic as the vLLM implementation: a lightweight
    speculator proposes a few look-ahead tokens and we compute the attention
    scores between those speculative queries and the base model keys to select
    important context tokens.
    """

    def __init__(
        self,
        base_model: str,
        spec_model: str,
        look_ahead: int = 4,
        keep_percentage: float = 0.5,
        pool_kernel_size: int = 1,
        device: str = "cuda",
    ) -> None:
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model).to(device)
        self.spec_model = AutoModelForCausalLM.from_pretrained(spec_model).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.spec_tokenizer = AutoTokenizer.from_pretrained(spec_model)
        self.look_ahead = look_ahead
        self.keep_percentage = keep_percentage
        self.pool_kernel_size = pool_kernel_size
        self.device = device
        self.base_model.eval()
        self.spec_model.eval()

    def _rotary_emb(self, position: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=self.device).float() / dim))
        freqs = torch.outer(torch.tensor([position], device=self.device).float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return cos, sin

    @torch.no_grad()
    def __call__(self, prompt: str) -> torch.Tensor:
        base_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        spec_ids = self.spec_tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        # 1. run base model to collect key cache from context
        base_out = self.base_model(
            base_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        base_keys = [k.squeeze(0) for k, _ in base_out.past_key_values]

        seq_len = base_ids.size(1)

        # 2. look-ahead speculation using speculator model
        spec_cache = None
        query_buffer: List[List[torch.Tensor]] = [[] for _ in range(len(base_keys))]
        new_tokens: List[torch.Tensor] = []
        eos_id = self.tokenizer.eos_token_id

        for step in range(self.look_ahead):
            outputs = self.spec_model(
                spec_ids if spec_cache is None else spec_ids[:, -1:],
                use_cache=True,
                past_key_values=spec_cache,
                output_hidden_states=True,
                return_dict=True,
            )

            spec_cache = outputs.past_key_values
            hidden = [h[:, -1, :] for h in outputs.hidden_states[1:]]

            for layer_idx, h in enumerate(hidden):
                attn = self.spec_model.model.layers[layer_idx].self_attn
                cos, sin = self._rotary_emb(seq_len + step, attn.head_dim)
                q = attn.q_proj(h)
                k = attn.k_proj(h)
                q = q.view(1, attn.num_attention_heads, 1, attn.head_dim)
                k = k.view(1, attn.num_key_value_heads, 1, attn.head_dim)
                q, _ = apply_rotary_pos_emb(q, k, cos, sin)
                query_buffer[layer_idx].append(q.squeeze(2))

            next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
            new_tokens.append(next_token)
            spec_ids = torch.cat([spec_ids, next_token], dim=1)
            if next_token.item() == eos_id:
                break

        queries = [torch.cat(qs, dim=1) for qs in query_buffer]
        keys = [k.transpose(0, 1) for k in base_keys]

        attn_scores = []
        for q_layer, k_layer in zip(queries, keys):
            # q_layer: [num_heads, steps, head_dim]
            # k_layer: [num_heads, seq_len, head_dim]
            attn = torch.einsum("hsd,hld->hsl", q_layer, k_layer) / (q_layer.size(-1) ** 0.5)
            attn_scores.append(attn)

        attn = torch.stack(attn_scores)  # [layer, head, step, seq]
        attn = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(attn.dtype)
        attn = attn.flatten(0, 1)
        if self.pool_kernel_size > 1:
            attn = torch.nn.functional.avg_pool1d(
                attn,
                kernel_size=self.pool_kernel_size,
                stride=1,
                padding=self.pool_kernel_size // 2,
            )
        attn = attn.max(0)[0]
        token_importance = attn.mean(0)

        topk = max(1, int(seq_len * self.keep_percentage))
        keep_idx = torch.topk(token_importance, k=topk).indices.sort().values
        kept_ids = base_ids[:, keep_idx]

        final_ids = torch.cat([kept_ids] + new_tokens, dim=1)
        return self.base_model(final_ids).logits
