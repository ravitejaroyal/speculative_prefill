import math
import os
from abc import abstractmethod
from typing import Dict, Optional, Tuple

import torch
from transformers import LlamaForCausalLM, LlamaModel
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.sequence import ExecuteModelRequest

from speculative_prefill.vllm_patch.config import SpecConfig
from speculative_prefill.vllm_patch.data.sequence import AugmentedSequenceData
from speculative_prefill.vllm_patch.models.llama import (
    enable_fa2_output_attns, visualize_attns)


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
        for metadata in execute_model_request.seq_group_metadata_list:
            if metadata.is_prompt:
                cur_indices = all_indices[index_pos].cpu()
                index_pos += 1

                # get the seq data
                assert len(metadata.seq_data) == 1
                seq_id = metadata.get_first_seq_id()
                seq_data = metadata.seq_data[seq_id]
                
                prompt_token_ids = seq_data._prompt_token_ids
                output_token_ids = seq_data._output_token_ids
                
                new_seq_data = AugmentedSequenceData.from_seqs_and_pos_ids(
                    prompt_token_ids=torch.LongTensor(prompt_token_ids)[cur_indices], 
                    position_ids=cur_indices.tolist(), 
                    output_token_ids=output_token_ids
                )

                metadata.seq_data[seq_id] = new_seq_data

            new_seq_group_metadata_list.append(metadata)

        assert index_pos == len(all_indices)
        return execute_model_request.clone(
            seq_group_metadata_list=new_seq_group_metadata_list)
    
    @abstractmethod
    def _speculate_indices(
        self, 
        execute_model_req: ExecuteModelRequest | None = None
    ) -> Tuple[torch.LongTensor]:
        raise NotImplementedError

    def init_device(self):
        pass

    def load_model(self):
        pass


class HFSpecWorker(SpecWorker):
    def __init__(
        self, 
        spec_model_name: str
    ) -> None:
        self.spec_model_name = spec_model_name
        self.model: Optional[LlamaForCausalLM] = None

        self.spec_config = SpecConfig.from_path(
            os.environ.get("SPEC_CONFIG_PATH", None)
        )

        print("\033[92m{}\033[00m".format(
            f"Using spec config:\n{self.spec_config}"))

    def load_model(self):
        if get_tensor_model_parallel_rank() == 0:
            model_cls = LlamaModel \
                if self.spec_config.algo == "attn" else LlamaForCausalLM
            self.model = model_cls.from_pretrained(
                self.spec_model_name, 
                torch_dtype=torch.bfloat16, 
                device_map="auto", 
                attn_implementation="flash_attention_2",
            )

            for param in self.model.parameters():
                param.requires_grad_(False)

            if self.spec_config.algo.startswith("attn"):
                # patch for enabling attn outputs                
                self.model = enable_fa2_output_attns(self.model, self.spec_config)
            else:
                if self.spec_config.gradient_checkpointing and \
                not self.model.model.gradient_checkpointing:
                    self.model.gradient_checkpointing_enable()
                    self.model.model.training = True

    def _speculate_indices(
        self, 
        execute_model_req: ExecuteModelRequest | None = None
    ) -> Tuple[torch.LongTensor]:
        hf_kwargs = self._extract_hf_inputs(execute_model_req)
        
        token_importance = getattr(
            self, 
            f"_compute_token_importance_by_{self.spec_config.algo}"
        )(**hf_kwargs)

        # for each sample we choose the indices
        kept_indices = ()
        for sample_ti in token_importance:
            seq_len = len(sample_ti)
            if self.spec_config.keep_strategy == "percentage":
                topk = math.ceil(seq_len * self.spec_config.keep_kwargs["percentage"])
            elif self.spec_config.keep_strategy == "constant":
                topk = min(seq_len, self.spec_config.keep_kwargs["constant"])
            elif self.spec_config.keep_strategy == "percentage-lowerbound":
                topk = math.ceil(seq_len * self.spec_config.keep_kwargs["percentage"])
                topk = max(topk, self.spec_config.keep_kwargs["lowerbound"])
            _, indices = torch.topk(sample_ti, k=topk, dim=-1)
            kept_indices = kept_indices + (torch.sort(indices)[0], )

        return kept_indices

    @torch.inference_mode
    def _compute_token_importance_by_attn(
        self, 
        **hf_kwargs
    ) -> Tuple[torch.Tensor]:
        hf_kwargs.pop("seq_lens")
        hf_kwargs.pop("last_token_pos")

        # use the base model to avoid the LM head
        output: BaseModelOutputWithPast = self.model.forward(
            hf_kwargs.pop("input_ids"), 
            use_cache=False, 
            output_attentions=True,             
            return_dict=True, 
            **hf_kwargs
        )

        attn_weights = output.attentions
        
        if self.spec_config.algo_kwargs.get("visualize", False):
            visualize_attns(
                attns=attn_weights, 
                **self.spec_config.algo_kwargs
            )

        agg_attns = []

        layer_start = self.spec_config.algo_kwargs.get("layer_start", None)
        layer_end = self.spec_config.algo_kwargs.get("layer_end", None)

        for sample_idx in range(len(attn_weights[0])):
            # [num of layers, head, 1, seqlen]
            all_layer_attns = torch.concatenate([aw[sample_idx] for aw in attn_weights], dim=0)
            # take of layers if necessary
            all_layer_attns = all_layer_attns[layer_start:layer_end]
            # max over heads
            all_layer_attns = all_layer_attns.max(1)[0]
            # smooth out attn
            kernel_size = self.spec_config.algo_kwargs.get("pool_kernel_size", None)
            if kernel_size:
                all_layer_attns = torch.nn.functional.avg_pool1d(
                    all_layer_attns, 
                    kernel_size=kernel_size, 
                    padding=kernel_size // 2,
                    stride=1
                )
            # average over layers
            merge_fn = self.spec_config.algo_kwargs.get("merge_fn", "max")
            if merge_fn == "max":
                attns = all_layer_attns.max(0)[0].view(-1)
            elif merge_fn == "sum":
                attns = all_layer_attns.sum(0).view(-1)
            elif merge_fn == "mean":
                attns = all_layer_attns.mean(0).view(-1)
            else:
                raise ValueError
            
            agg_attns.append(attns)

        return tuple(agg_attns)

    @torch.inference_mode
    def _compute_token_importance_by_attn_lah(
         self, 
        **hf_kwargs
    ) -> Tuple[torch.Tensor]:
        seq_lens = hf_kwargs.pop("seq_lens")
        _ = hf_kwargs.pop("last_token_pos")   
        
        assert len(seq_lens) == 1, "Currently look ahead only supports bs = 1."
        
        input_ids = hf_kwargs.pop("input_ids")
        output: GenerateDecoderOnlyOutput = self.model.generate(
            input_ids=input_ids, 
            attention_mask=torch.ones_like(input_ids), 
            use_cache=True, 
            output_attentions=True,             
            return_dict_in_generate=True, 
            return_legacy_cache=False, 
            do_sample=False, 
            eos_token_id=128009, 
            pad_token_id=128009, 
            temperature=None, 
            top_p=None, 
            max_new_tokens=self.spec_config.algo_kwargs.get("lah_cnt", 4)
        )

        all_attns = []
        prefill_len = seq_lens[0]
        attentions = output.attentions

        layer_start = self.spec_config.algo_kwargs.get("layer_start", None)
        layer_end = self.spec_config.algo_kwargs.get("layer_end", None)

        for token_pos in range(len(attentions)):
            # Tuple[B, H, 1, K]
            attn = attentions[token_pos]
            # [layer_cnt, B, H, 1, K]
            attn = torch.stack(attn, dim=0)[layer_start:layer_end]
            # [layer_cnt, B, prefill_len]
            attn = attn.max(dim=2)[0][..., :prefill_len].squeeze(-2).squeeze(-2)
            # normalize
            attn = torch.nn.functional.softmax(attn, dim=-1)
            # smooth it out
            kernel_size = self.spec_config.algo_kwargs.get("pool_kernel_size", None)
            if kernel_size:
                attn = torch.nn.functional.avg_pool1d(
                    attn, 
                    kernel_size=kernel_size, 
                    padding=kernel_size // 2,
                    stride=1
                )
            # max over layers
            attn = torch.max(attn, dim=0)[0]
            # aggregate
            all_attns.append(attn)

        # aggregate attns over look ahead tokens
        if self.spec_config.algo_kwargs.get("lah_moving_avg", None) is not None:
            scores = torch.zeros_like(all_attns[0])
            factor = self.spec_config.algo_kwargs["lah_moving_avg"]
            for attn in reversed(all_attns):
                scores = scores * (1 - factor) + attn * factor
        else:
            scores = torch.stack(all_attns, dim=0).mean(0)

        return (scores, )

    def _compute_token_importance_by_backprop(
        self, 
        **hf_kwargs
    ) -> Tuple[torch.Tensor]:
        seq_lens = hf_kwargs.pop("seq_lens")
        last_token_pos = hf_kwargs.pop("last_token_pos")

        with torch.enable_grad():
            _inputs_embeds: torch.Tensor = self.model.model.embed_tokens(
                hf_kwargs.pop("input_ids")
            )

            if self.spec_config.algo_kwargs.get("use_sub_space", -1) > 0:
                sub_inputs_embeds = torch.split(_inputs_embeds, [
                    self.spec_config.use_sub_space, 
                    self.model.config.hidden_size - self.spec_config.use_sub_space
                ], dim=-1)
                sub_inputs_embeds[0].requires_grad_(True)
                sub_inputs_embeds[0].retain_grad()
                inputs_embeds = torch.concat(sub_inputs_embeds, dim=-1)
            else:
                _inputs_embeds.requires_grad_(True)
                _inputs_embeds.retain_grad()
                inputs_embeds = _inputs_embeds

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
            if self.spec_config.algo_kwargs.get("use_sub_space", -1) > 0:
                grads = sub_inputs_embeds[0].grad
            else:
                grads = inputs_embeds.grad

        # [total_token_cnt]
        grad_magnitudes = torch.linalg.vector_norm(grads, dim=-1).view(-1)

        # Tuple [seqlen]
        grad_magnitudes = torch.split(grad_magnitudes, seq_lens.tolist())

        return grad_magnitudes

    def _extract_hf_inputs(
        self, 
        execute_model_req: ExecuteModelRequest | None = None
    ) -> Dict[str, torch.Tensor]:
        input_ids = []
        position_ids = []
        seq_lens = []

        for metadata in execute_model_req.seq_group_metadata_list:
            if metadata.is_prompt:
                assert len(metadata.seq_data) == 1
                seq_id = metadata.get_first_seq_id()
                prompt_token_ids = metadata.seq_data[seq_id].prompt_token_ids
                prompt_len = len(prompt_token_ids)

                input_ids.extend(prompt_token_ids)
                position_ids.append(torch.arange(0, prompt_len))
                seq_lens.append(prompt_len)

        input_ids = torch.LongTensor(input_ids)
        position_ids = torch.concatenate(position_ids).to(torch.int64)
        seq_lens = torch.LongTensor(seq_lens)
        last_token_pos = torch.cumsum(seq_lens, dim=-1) - 1

        return {
            "input_ids": input_ids.unsqueeze(0).to(self.model.device), 
            "position_ids": position_ids.unsqueeze(0).cuda(self.model.device), 
            "seq_lens": seq_lens, 
            "last_token_pos": last_token_pos
        }
