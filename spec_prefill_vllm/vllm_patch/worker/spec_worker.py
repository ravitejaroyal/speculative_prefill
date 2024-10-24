import math
import os
from abc import abstractmethod
from typing import Dict, Tuple

import torch
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from vllm.sequence import ExecuteModelRequest
from vllm_patch.config import SpecConfig
from vllm_patch.data.sequence import AugmentedSequenceData


class SpecWorker:
    def speculate(
        self, 
        execute_model_request: ExecuteModelRequest
    ) -> ExecuteModelRequest:
        # Don't do anything if all requests are decode
        if sum([data.is_prompt for data 
            in execute_model_request.seq_group_metadata_list]) == 0:
            return execute_model_request

        all_indices = self._speculate_indices(execute_model_request)
        # now we need to update the original request
        index_pos = 0
        new_seq_group_metadata_list = []
        for seq_group_metadata in execute_model_request.seq_group_metadata_list:
            if seq_group_metadata.is_prompt:
                cur_indices = all_indices[index_pos].cpu()
                index_pos += 1

                # get the seq data
                request_id = int(seq_group_metadata.request_id)
                seq_data = seq_group_metadata.seq_data[request_id]
                
                prompt_token_ids = seq_data._prompt_token_ids
                output_token_ids = seq_data._output_token_ids
                
                new_seq_data = AugmentedSequenceData.from_seqs_and_pos_ids(
                    prompt_token_ids=torch.LongTensor(prompt_token_ids)[cur_indices], 
                    position_ids=cur_indices.tolist(), 
                    output_token_ids=output_token_ids
                )

                seq_group_metadata.seq_data[request_id] = new_seq_data

            new_seq_group_metadata_list.append(seq_group_metadata)

        assert index_pos == len(all_indices)
        return execute_model_request.clone(
            seq_group_metadata_list=new_seq_group_metadata_list)
    
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

        self.spec_config = SpecConfig.from_path(
            os.environ.get("spec_config_path", None)
        )

        print(f"Using spec config:\n{self.spec_config}")

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
            assert self.spec_config.keep_strategy == "percentage"
            topk = math.ceil(seq_len * self.spec_config.keep_kwargs["percentage"])
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
            if seq_group_metadata.is_prompt:
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
