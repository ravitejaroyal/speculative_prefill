from copy import deepcopy
from typing import List, Optional, Tuple

import torch
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest, SequenceGroupMetadata
from vllm.worker.worker import Worker


class LookAheadSpecWorker(Worker):
    @torch.inference_mode
    def speculate(
        self, 
        execute_model_req: ExecuteModelRequest, 
        look_ahead_cnt: int = 1
    ) -> ExecuteModelRequest:
        
        self._raise_if_unsupported(execute_model_req)

        request, non_empty = self._extract_prompt_execute_model_req(
            execute_model_req=execute_model_req, 
            look_ahead_cnt=look_ahead_cnt
        )

        if not non_empty:
            return execute_model_req

        prefill_slot_mapping = None
        decode_slot_mapping = None

        model_outputs: List[SamplerOutput] = []

        for itr in range(look_ahead_cnt):
            prefill_slot_mapping, decode_slot_mapping = self._update_slot_mapping(
                request, 
                prefill_slot_mapping=prefill_slot_mapping, 
                decode_slot_mapping=decode_slot_mapping
            )

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
            decode_slot_mapping=decode_slot_mapping
        )
        
        exit()

        # # get the attention score somehow
        # assert self.kv_cache is not None
        # # layer_cnt, (2, num_blocks, block_size, num_kv_heads, head_size)
        # kv_caches = self.kv_cache[execute_model_req.virtual_engine]
        
        # num_hidden_layers = getattr(
        #     self.model_runner.model_config.hf_text_config, "num_hidden_layers")
        

    def _get_attention_scores(
        self, 
        prefill_slot_mapping: Optional[List[torch.Tensor]], 
        decode_slot_mapping: Optional[List[torch.Tensor]]
    ):
        pass

    def _get_tensors_from_slot_mapping(
        slot_mapping: torch.Tensor, 
        cache_pos: int
    ):
        """
            cache_pos = 0 means keys
            cache_pos = 1 means values
        """
        assert cache_pos in [0, 1]
        pass

    def _update_slot_mapping(
        self, 
        execute_model_req: ExecuteModelRequest, 
        prefill_slot_mapping: Optional[List[torch.Tensor]], 
        decode_slot_mapping: Optional[List[torch.Tensor]]
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        model_input, _,  _ = self.prepare_input(
            execute_model_req
        )

        self.model_runner.attn_state.begin_forward(model_input)
        assert model_input.attn_metadata is not None
        prefill_metadata: FlashAttentionMetadata = model_input.attn_metadata.prefill_metadata
        decode_metadata: FlashAttentionMetadata = model_input.attn_metadata.decode_metadata

        # exclusive check
        assert bool(prefill_metadata is not None) != bool(decode_metadata is not None)

        if prefill_metadata is not None:
            return list(torch.split(
                prefill_metadata.slot_mapping, 
                prefill_metadata.seq_lens_tensor.tolist())), decode_slot_mapping
        else:
            assert decode_metadata.decode_metadata == 1
            assert decode_metadata.num_decode_tokens == len(decode_metadata.slot_mapping)

            if decode_slot_mapping == None:
                decode_slot_mapping = list(torch.split(decode_metadata.slot_mapping, 1))
            else:
                for sample_idx in range(len(decode_slot_mapping)):
                    decode_slot_mapping[sample_idx] = torch.concat([
                            decode_slot_mapping[sample_idx], 
                            decode_metadata.slot_mapping[sample_idx: sample_idx + 1]
                        ], dim=0)

            return prefill_slot_mapping, decode_slot_mapping

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

    