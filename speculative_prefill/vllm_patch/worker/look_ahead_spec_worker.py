import math
from copy import deepcopy
from functools import partial
from types import MethodType
from typing import List, Optional, Tuple

import torch
from vllm.attention import AttentionMetadata
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.models.llama import (LlamaAttention,
                                              LlamaDecoderLayer,
                                              LlamaForCausalLM)
from vllm.sequence import ExecuteModelRequest, SequenceGroupMetadata
from vllm.worker.worker import Worker


def _forward_with_query_dump(
    self: LlamaAttention,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: AttentionMetadata, 
    query_buffer: List[List[torch.Tensor]], 
    layer_idx: int
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = self.rotary_emb(positions, q, k)

    if attn_metadata.num_prefills > 0:
        query_buffer[layer_idx].append(
            q[attn_metadata.seq_lens_tensor - 1, :])
    else:
        query_buffer[layer_idx].append(q)

    attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
    output, _ = self.o_proj(attn_output)
    return output


class LookAheadSpecWorker(Worker):
    def load_model(self):
        super().load_model()
        assert isinstance(self.model_runner.model, LlamaForCausalLM)
        
        self._prepare_query_buffer()

        # do patching here to allow model output attention
        for layer_idx, layer in enumerate(self.model_runner.model.model.layers):
            assert isinstance(layer, LlamaDecoderLayer)
            layer.self_attn.forward = MethodType(
                partial(
                    _forward_with_query_dump, 
                    query_buffer=self.query_buffer, 
                    layer_idx=layer_idx), 
                layer.self_attn)
            
        self._reshape_key = partial(
            torch.repeat_interleave, 
            dim=1, 
            repeats=(
                self.model_runner.model.config.num_attention_heads //
                self.model_runner.model.config.num_key_value_heads
            ))
        
        self._reshape_query = lambda x: x.reshape(
            -1, 
            self.model_runner.model.config.num_attention_heads, 
            self.model_runner.model.config.head_dim
        )

    def _prepare_query_buffer(self):
        if not hasattr(self, "query_buffer"):
            self.query_buffer = [
                [] for _ in range(self._get_model_num_layers())
            ]
        else:
            # using clear instead of resetting prevents losing ref
            for qb in self.query_buffer:
                qb.clear()

    @torch.inference_mode
    def speculate(
        self, 
        execute_model_req: ExecuteModelRequest, 
        look_ahead_cnt: int = 1
    ) -> ExecuteModelRequest:
        
        assert look_ahead_cnt > 0

        self._raise_if_unsupported(execute_model_req)

        request, non_empty = self._extract_prompt_execute_model_req(
            execute_model_req=execute_model_req, 
            look_ahead_cnt=look_ahead_cnt
        )

        if not non_empty:
            return execute_model_req

        self._prepare_query_buffer()

        model_outputs: List[SamplerOutput] = []

        for itr in range(look_ahead_cnt):
            if itr == 0:
                prefill_slot_mapping = self._get_prefill_slot_mapping(request)

            model_output: List[SamplerOutput] = super().execute_model(
                execute_model_req=request)
            assert (len(model_output) == 1), \
                "composing multistep workers not supported"
            model_output = model_output[0]

            self._append_new_tokens(
                model_output, 
                request.seq_group_metadata_list, 
                itr
            )
            model_outputs.append(model_output)

        self._get_attention_scores(
            prefill_slot_mapping=prefill_slot_mapping, 
            kv_cache=self.kv_cache[execute_model_req.virtual_engine]
        )
        
        exit()

        # # get the attention score somehow
        # assert self.kv_cache is not None
        # # layer_cnt, (2, num_blocks, block_size, num_kv_heads, head_size)
        # kv_caches = self.kv_cache[execute_model_req.virtual_engine]
        
        # num_hidden_layers = getattr(
        #     self.model_runner.model_config.hf_text_config, "num_hidden_layers")
        
    def _get_model_num_layers(self) -> int:
        return getattr(
            self.model_runner.model_config.hf_text_config, "num_hidden_layers")

    def _get_attention_scores(
        self, 
        prefill_slot_mapping: Optional[List[torch.Tensor]], 
        kv_cache: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
            A caveat or potentially a good design here:
                For look ahead tokens, we also just attend to context, 
                not generated tokens
        """

        attn_weights = []

        # first lets get the queries out from query_buffer
        for layer_idx in range(len(self.query_buffer)):
            attn_weights.append([])

            # get keys from kv cache (one tensor per sample)
            keys = self._get_keys_from_slot_mapping(
                slot_mapping=prefill_slot_mapping, 
                layer_idx=layer_idx, 
                kv_cache=kv_cache)
            
            # get queries from dump buffer [look_ahead_cnt, sample_cnt, tensors]
            queries = self.query_buffer[layer_idx]
            queries = torch.split(
                torch.stack(queries, dim=0), 
                split_size_or_sections=1, 
                dim=1)
            
            for q, k in zip(queries, keys):
                # (len, num_heads, head_dim)
                query = self._reshape_query(q).transpose(0, 1)
                key = self._reshape_key(k).transpose(0, 1)
                
                attn = torch.matmul(
                    query, key.transpose(-1, -2)
                ) / math.sqrt(query.shape[-1])

                attn = torch.nn.functional.softmax(
                    attn, 
                    dim=-1, 
                    dtype=torch.float32
                ).to(key.dtype)

                attn_weights[-1].append(attn)

        # reshape -> from [layer, sample, attn] to [sample, layer, attn]
        reshaped_attn_weights: List[torch.Tensor] = []
        for sample_idx in range(len(attn_weights[0])):
            reshaped_attn_weights.append(
                torch.stack(
                    [attn_weights[layer_idx][sample_idx] for layer_idx in range(len(attn_weights))], 
                    dim=0))

        return reshaped_attn_weights

    def _get_keys_from_slot_mapping(
        self, 
        slot_mapping: List[torch.Tensor], 
        layer_idx: int, 
        kv_cache: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        # (num_blocks, block_size, num_kv_heads, head_size)
        key_cache = kv_cache[layer_idx][0]
        block_size = key_cache.shape[1]

        keys = []
        
        for sm in slot_mapping:
            block_indices = sm // block_size
            block_pos = sm % block_size

            keys.append(key_cache[block_indices, block_pos, :, :])
            
        return keys

    def _get_prefill_slot_mapping(
        self, 
        execute_model_req: ExecuteModelRequest, 
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        model_input, _,  _ = self.prepare_input(
            execute_model_req
        )

        self.model_runner.attn_state.begin_forward(model_input)
        assert model_input.attn_metadata is not None
        prefill_metadata: FlashAttentionMetadata = model_input.attn_metadata.prefill_metadata
        assert prefill_metadata is not None

        return list(torch.split(
            prefill_metadata.slot_mapping, 
            prefill_metadata.seq_lens_tensor.tolist()))

    def _extract_prompt_execute_model_req(
        self, 
        execute_model_req: ExecuteModelRequest, 
        look_ahead_cnt: int
    ) -> Tuple[ExecuteModelRequest, bool]:
        prompt_seq_group_metadata_list = [
            deepcopy(sgm) for sgm in execute_model_req.seq_group_metadata_list \
            if sgm.is_prompt
        ]

        request = execute_model_req.clone(
            seq_group_metadata_list=prompt_seq_group_metadata_list, 
        )

        request.num_lookahead_slots = look_ahead_cnt

        return request, len(prompt_seq_group_metadata_list) > 0

    def _append_new_tokens(
        self,
        model_output: List[SamplerOutput],
        seq_group_metadata_list: List[SequenceGroupMetadata], 
        itr: int
    ) -> None:
        """Given model output from a single run, append the tokens to the
        sequences. This is normally done outside of the worker, but it is
        required if the worker is to perform multiple forward passes.
        """
        for seq_group_metadata, sequence_group_outputs in zip(
                seq_group_metadata_list, model_output):
            seq_group_metadata.is_prompt = False

            for seq_output in sequence_group_outputs.samples:
                # NOTE: Beam search is not supported, so we can assume that
                # parent_seq_id == seq_id.
                seq = seq_group_metadata.seq_data[seq_output.parent_seq_id]

                token_id = seq_output.output_token
                token_logprob = seq_output.logprobs[token_id]

                seq.append_token_id(token_id, token_logprob.logprob)
                if itr == 0:
                    # prefill
                    new_token = seq.get_prompt_len()
                else:
                    # decode
                    new_token = 1
                seq.update_num_computed_tokens(new_token)

    def _raise_if_unsupported(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> None:
        if any([
                execute_model_req.blocks_to_swap_in,
                execute_model_req.blocks_to_swap_out,
                execute_model_req.blocks_to_copy
        ]):
            raise NotImplementedError(
                "MultiStepWorker does not support cache operations")

        if any(
                len(seq_group_metadata.seq_data.keys()) != 1
                for seq_group_metadata in
                execute_model_req.seq_group_metadata_list):
            raise NotImplementedError(
                "MultiStepWorker does not support beam search.")
