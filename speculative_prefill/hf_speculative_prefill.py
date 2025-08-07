import math
import torch
from torch.nn import functional as F
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import WEIGHTS_NAME
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)

torch.backends.cuda.enable_flash_sdp(True)  # use Flash-Attention-2 kernels
dtype = torch.float16  # keep AWQ de-quant in fp16


class HFFSpeculativePrefill:
    """Standalone speculative prefill implementation using HuggingFace models."""

    def __init__(
        self,
        base_model: str,
        spec_model: str,
        look_ahead: int = 4,
        keep_percentage: float = 0.5,
        *,
        pool_kernel_size: Optional[int] = None,
        chunk_size: Optional[int] = None,
        tensor_parallel: Optional[bool] = None,
        device: str = "cuda",
        speculative: bool = True,
    ) -> None:
        self.look_ahead = look_ahead
        self.keep_percentage = keep_percentage
        self.pool_kernel_size = pool_kernel_size
        self.chunk_size = chunk_size
        self.tensor_parallel = tensor_parallel
        self.speculative = speculative

        if device.startswith("cuda"):
            try:
                from autoawq import AutoAWQForCausalLM
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "autoawq is required to load AWQ models"
                ) from exc

            base = AutoAWQForCausalLM.from_quantized(
                base_model,
                device_map="auto",
                max_memory={0: "72GiB", 1: "72GiB", "cpu": "4GiB"},
                fuse_layers=True,
                use_flash_attention_2=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            self.base_model = base
            self.tokenizer = tokenizer
            self.device = "cuda:0"  # prompt accumulators stay on GPU-0
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(base_model).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            self.device = device

        self.base_model.eval()

        if self.speculative:
            if device.startswith("cuda"):
                spec = AutoModelForCausalLM.from_pretrained(
                    spec_model,
                    device_map={"": 1},
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                )
            else:
                spec = AutoModelForCausalLM.from_pretrained(spec_model).to(device)
            self.spec_model = spec
            self.spec_tokenizer = AutoTokenizer.from_pretrained(spec_model)
            self.spec_device = next(self.spec_model.parameters()).device
            self.spec_model.eval()
            self._patch_speculator_attn()
        else:
            self.spec_model = None
            self.spec_tokenizer = None
            self.spec_device = device

    def _load_prompt(self, prompt: Union[str, Path]) -> str:
        """Return prompt text, reading from file if ``prompt`` is a path."""
        p = Path(prompt)
        if p.exists():
            return p.read_text().strip()
        return str(prompt)

    @torch.no_grad()
    def __call__(self, prompt: Union[str, Path]) -> torch.Tensor:
        text = self._load_prompt(prompt)
        base_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(
            self.device
        )
        seq_len = base_ids.size(1)
        steps = self.look_ahead

        if not self.speculative:
            pos_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
            self.last_final_ids = base_ids
            self.last_position_ids = pos_ids
            return self.base_model(base_ids, position_ids=pos_ids).logits

        spec_ids = self.spec_tokenizer(text, return_tensors="pt").input_ids.to(
            self.spec_device
        )

        # 1. run speculator on context to build key cache
        spec_out = self.spec_model(spec_ids, use_cache=True, return_dict=True)
        spec_cache = spec_out.past_key_values
        spec_keys = [k.squeeze(0) for k, _ in spec_cache]
        layer_queries = [qbuf[-1] for qbuf in self.query_buffer]
        queries_per_step: List[torch.Tensor] = [
            torch.stack(layer_queries, dim=0)
        ]
        self._clear_query_buffer()

        if steps == 0:
            pos_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
            self.last_final_ids = base_ids
            self.last_position_ids = pos_ids
            return self.base_model(base_ids, position_ids=pos_ids).logits

        outputs = spec_out
        new_tokens: List[torch.Tensor] = []
        for _ in range(steps):
            next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
            outputs = self.spec_model(
                next_token,
                use_cache=True,
                past_key_values=spec_cache,
                return_dict=True,
            )
            spec_cache = outputs.past_key_values
            layer_queries = [qbuf[-1] for qbuf in self.query_buffer]
            queries_per_step.append(torch.stack(layer_queries, dim=0))
            self._clear_query_buffer()
            spec_ids = torch.cat([spec_ids, next_token], dim=1)
            token_id = next_token.item()
            text = self.spec_tokenizer.decode([token_id], skip_special_tokens=True)
            mapped = (
                self.tokenizer(
                    text, add_special_tokens=False, return_tensors="pt"
                )
                .input_ids.to(self.device)
            ).long()
            new_tokens.append(mapped)
            if token_id == self.spec_tokenizer.eos_token_id:
                break

        # 2. compute attention scores between speculator queries and keys
        queries = torch.stack(queries_per_step, dim=0).permute(1, 2, 0, 3)
        is_tp = self._is_tensor_parallel()
        if is_tp:
            queries = self._gather_head_dim(queries)
        keys = torch.stack(spec_keys, dim=0)
        repeat_factor = (
            self.spec_model.config.num_attention_heads
            // self.spec_model.config.num_key_value_heads
        )
        keys = keys.repeat_interleave(repeat_factor, dim=1)
        if is_tp:
            keys = self._gather_head_dim(keys)

        attn = torch.einsum("lhsd, lhnd -> lhsn", queries, keys) / math.sqrt(
            queries.shape[-1]
        )
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(queries.dtype)
        attn = attn.flatten(0, 1)  # [layer*head, step, seq]

        if self.pool_kernel_size and self.pool_kernel_size > 1:
            attn = F.avg_pool1d(
                attn,
                kernel_size=self.pool_kernel_size,
                padding=self.pool_kernel_size // 2,
                stride=1,
            )

        attn = attn.max(dim=0)[0]  # [step, seq]
        token_importance = attn.mean(dim=0)  # [seq]
        token_importance = token_importance.to(self.device)

        # 4. select important tokens optionally using chunking
        if self.chunk_size and self.chunk_size > 0:
            chunks = token_importance.split(self.chunk_size, dim=-1)
            chunk_scores = torch.stack([c.mean() for c in chunks])
            keep_chunks = max(1, math.ceil(len(chunks) * self.keep_percentage))
            _, chunk_idx = torch.topk(chunk_scores, k=keep_chunks)
            indices = torch.cat(
                [
                    torch.arange(
                        i * self.chunk_size,
                        min((i + 1) * self.chunk_size, seq_len),
                        device=self.device,
                    )
                    for i in chunk_idx.tolist()
                ]
            )
        else:
            k = max(1, math.ceil(seq_len * self.keep_percentage))
            _, indices = torch.topk(token_importance, k=k)

        keep_idx = indices.sort().values
        keep_idx = torch.unique(
            torch.cat(
                [keep_idx, torch.tensor([seq_len - 1], device=self.device)]
            )
        ).sort().values
        kept_ids = base_ids[:, keep_idx]

        # kept context tokens must precede newly generated tokens
        final_ids = torch.cat([kept_ids] + new_tokens, dim=1)
        new_len = sum(t.size(1) for t in new_tokens)
        pos_ids = torch.cat(
            [
                keep_idx.unsqueeze(0),
                torch.arange(
                    seq_len,
                    seq_len + new_len,
                    device=self.device,
                ).unsqueeze(0),
            ],
            dim=1,
        )
        self.last_final_ids = final_ids
        self.last_position_ids = pos_ids
        pruned_text = self.tokenizer.decode(final_ids[0], skip_special_tokens=True)
        print(f"Speculative prefill output: {pruned_text}")
        return self.base_model(final_ids, position_ids=pos_ids).logits

    def _is_tensor_parallel(self) -> bool:
        if self.tensor_parallel is not None:
            return self.tensor_parallel
        return (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )

    def _gather_head_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self._is_tensor_parallel():
            return tensor
        world_size = torch.distributed.get_world_size()
        parts = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(parts, tensor)
        return torch.cat(parts, dim=1)

    def _patch_speculator_attn(self) -> None:
        self.query_buffer: List[List[torch.Tensor]] = [
            [] for _ in range(self.spec_model.config.num_hidden_layers)
        ]
        for layer_idx, layer in enumerate(self.spec_model.model.layers):
            attn = layer.self_attn

            def forward(
                self_attn,
                hidden_states: torch.Tensor,
                position_embeddings: tuple,
                attention_mask: Optional[torch.Tensor],
                past_key_value: Optional[tuple] = None,
                cache_position: Optional[torch.LongTensor] = None,
                _layer_idx: int = layer_idx,
                **kwargs,
            ):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self_attn.head_dim)

                q = self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                k = self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                v = self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

                if past_key_value is not None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    k, v = past_key_value.update(k, v, self_attn.layer_idx, cache_kwargs)

                self.query_buffer[_layer_idx].append(q[0, :, -1, :])

                attention_interface = eager_attention_forward
                if self_attn.config._attn_implementation != "eager":
                    attention_interface = ALL_ATTENTION_FUNCTIONS[
                        self_attn.config._attn_implementation
                    ]

                attn_output, attn_weights = attention_interface(
                    self_attn,
                    q,
                    k,
                    v,
                    attention_mask,
                    dropout=0.0 if not self_attn.training else self_attn.attention_dropout,
                    scaling=self_attn.scaling,
                    **kwargs,
                )

                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = self_attn.o_proj(attn_output)
                return attn_output, attn_weights

            attn.forward = forward.__get__(attn, type(attn))

    def _clear_query_buffer(self) -> None:
        for buf in self.query_buffer:
            buf.clear()

    def measure_ttft(
        self, prompt: Union[str, Path], *, max_new_tokens: int = 16
    ) -> float:
        """Measure and print TTFT along with the model's full output.

        The timing covers the entire speculative or non-speculative prefill up to
        the base model's first token. Afterward, the base model continues to
        generate ``max_new_tokens`` more tokens starting from the prompt used for
        the prefill stage.
        """
        text = self._load_prompt(prompt)
        start = time.perf_counter()
        _ = self(text)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        ttft = time.perf_counter() - start
        label = "spec" if self.speculative else "no spec"
        print(f"TTFT ({label}): {ttft:.4f}s")

        gen_ids = self.base_model.generate(
            self.last_final_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        new_ids = gen_ids[0, self.last_final_ids.size(1) :]
        out_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        print(out_text)
        return ttft
