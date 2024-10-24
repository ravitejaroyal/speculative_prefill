import os
from typing import Dict, List, Tuple

import torch
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.worker.model_runner import ModelRunner
from vllm.worker.worker import Worker
from vllm.worker.worker_base import LoraNotSupportedWorkerBase

from vllm_patch.data.input_builder import AugmentedModelInputForGPUBuilder
from vllm_patch.data.sequence import AugmentedSequenceData
from vllm_patch.worker.spec_worker import HFSpecWorker, SpecWorker


def create_spec_worker(*args, **kwargs) -> "SpecPrefillWorker":
    spec_model_name = os.environ.get("spec_model", None)
    assert spec_model_name is not None

    assert kwargs["scheduler_config"].chunked_prefill_enabled == False, \
        "Please set --enable-chunked-prefill=False or enable_chunked_prefill=False. "

    # create the base model
    kwargs["model_runner_cls"] = ModelRunner
    base_model_worker = Worker(*args, **kwargs) 

    # create the spec prefill model
    spec_model_worker = HFSpecWorker(spec_model_name=spec_model_name)

    return SpecPrefillWorker(
        base_model_worker=base_model_worker, 
        spec_model_worker=spec_model_worker
    )


class SpecPrefillWorker(LoraNotSupportedWorkerBase):
    def __init__(
        self, 
        base_model_worker: Worker, 
        spec_model_worker: SpecWorker
    ):
        self.base_model_worker = base_model_worker
        self.spec_model_worker = spec_model_worker

        self.base_model_worker.model_runner._builder_cls = \
            AugmentedModelInputForGPUBuilder
        
        self.id_to_context_len: Dict[str, int] = {}

    def init_device(self) -> None:
        """Initialize device state, such as loading the model or other on-device
        memory allocations.
        """
        self.base_model_worker.init_device()
        self.base_model_worker.load_model()

        self.spec_model_worker.init_device()
        self.spec_model_worker.load_model()

    def load_model(self, *args, **kwargs):
        pass

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        return self.base_model_worker.determine_num_available_blocks()
        
    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.base_model_worker.initialize_cache(
            num_gpu_blocks=num_gpu_blocks, 
            num_cpu_blocks=num_cpu_blocks
        )
    
    # don't use inference mode which doesn't allow re-enabling grad compute
    @torch.no_grad
    def execute_model(
        self, 
        execute_model_req: ExecuteModelRequest | None = None
    ) -> List[SamplerOutput] | None:
        # TP > 1 will need this check
        if execute_model_req is not None:
            execute_model_req = self.spec_model_worker.speculate(execute_model_req)
            execute_model_req = self._record_and_update_requests(execute_model_req)
        return self.base_model_worker.execute_model(execute_model_req)

    def _record_and_update_requests(
        self, 
        execute_model_req: ExecuteModelRequest
    ) -> ExecuteModelRequest:
        for metadata in execute_model_req.seq_group_metadata_list:
            assert len(metadata.seq_data) == 1
            request_id = metadata.request_id
            seq_id = metadata.get_first_seq_id()
            id = f"{request_id}_{seq_id}"
            seq_data: AugmentedSequenceData = metadata.seq_data[seq_id]

            if metadata.is_prompt:    
                self.id_to_context_len[id] = seq_data.get_prompt_len()
            else:
                seq_data._context_len = self.id_to_context_len[id]
                metadata.seq_data[seq_id] = seq_data
                # we decode every time
                self.id_to_context_len[id] += 1

        return execute_model_req

    def get_cache_block_size_bytes(self) -> int:
        raise NotImplementedError