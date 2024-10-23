import math
from abc import abstractmethod
from typing import Dict, Tuple

import torch
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from vllm.sequence import ExecuteModelRequest


class SpecWorker:
    def speculate(
        self, 
        original_request: ExecuteModelRequest, 
        filtered_request: ExecuteModelRequest, 
    ) -> ExecuteModelRequest:
        indices = self._speculate_indices(filtered_request)
        # now we need to update the original request somehow
        

    
    @abstractmethod
    def _speculate_indices(
        self, 
        execute_model_req: ExecuteModelRequest | None = None
    ) -> Tuple[torch.LongTensor]:
        raise NotImplementedError


class HFSpecWorker(SpecWorker):
    def __init__(
        self, 
        spec_model_name: str
    ) -> None:
        self.model = LlamaForCausalLM.from_pretrained(
            spec_model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            attn_implementation="flash_attention_2",
        )

    def _speculate_indices(
        self, 
        execute_model_req: ExecuteModelRequest | None = None
    ) -> Tuple[torch.LongTensor]:
        hf_kwargs = self._extract_hf_inputs(execute_model_req)
        seq_lens = hf_kwargs.pop("seq_lens")
        last_token_pos = hf_kwargs.pop("last_token_pos")

        with torch.enable_grad():
            inputs_embeds: torch.Tensor = self.model.model.embed_tokens(
                hf_kwargs.pop("input_ids")
            )

            inputs_embeds.requires_grad_(True)
            inputs_embeds.retain_grad()

            output: CausalLMOutputWithPast = self.model.forward(
                inputs_embeds=inputs_embeds, 
                use_cache=False, 
                return_dict=True, 
                **hf_kwargs
            )

            # [1, total_token_cnt, vocab_size]
            logits = output.logits
            # take the logits of the last token of each sample (bs = 1 for packed)
            last_token_logits = logits[:, last_token_pos, :][0]
            # create noise
            noise = torch.randn_like(last_token_logits)
            noise = noise / torch.linalg.vector_norm(noise, dim=-1, keepdim=True)
            # target
            target = last_token_logits.detach() + noise
            # loss calculation
            loss = torch.nn.functional.cross_entropy(last_token_logits, target)
            # backward and get gradients
            loss.backward()
            # embeds grad
            grads = inputs_embeds.grad

        # [total_token_cnt]
        grad_magnitudes = torch.linalg.vector_norm(grads, dim=-1).view(-1)

        # Tuple [seqlen]
        grad_magnitudes = torch.split(grad_magnitudes, seq_lens.tolist())

        # for each sample we choose the indices
        kept_indices = ()
        for sample_gm in grad_magnitudes:
            seq_len = len(sample_gm)
            topk = math.ceil(seq_len * 0.6)
            _, indices = torch.topk(sample_gm, k=topk, dim=-1)
            kept_indices = kept_indices + (torch.sort(indices)[0], )

        return kept_indices
    
    def _extract_hf_inputs(
        self, 
        execute_model_req: ExecuteModelRequest | None = None
    ) -> Dict[str, torch.Tensor]:
        input_ids = []
        position_ids = []
        seq_lens = []

        for seq_group_metadata in execute_model_req.seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            request_id = seq_group_metadata.request_id
            prompt_token_ids = seq_group_metadata.seq_data[int(request_id)].prompt_token_ids
            prompt_len = len(prompt_token_ids)

            input_ids.extend(prompt_token_ids)
            position_ids.append(torch.arange(0, prompt_len))
            seq_lens.append(prompt_len)

        input_ids = torch.LongTensor(input_ids)
        position_ids = torch.concatenate(position_ids).to(torch.int64)
        seq_lens = torch.LongTensor(seq_lens)
        last_token_pos = torch.cumsum(seq_lens, dim=-1) - 1

        return {
            "input_ids": input_ids.unsqueeze(0), 
            "position_ids": position_ids.unsqueeze(0), 
            "seq_lens": seq_lens, 
            "last_token_pos": last_token_pos
        }
