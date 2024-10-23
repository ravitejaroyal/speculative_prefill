import os
from typing import List, Tuple

import torch
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.worker.model_runner import ModelRunner
from vllm.worker.worker import Worker
from vllm.worker.worker_base import LoraNotSupportedWorkerBase
from vllm_patch.worker.spec_worker_base import HFSpecWorker, SpecWorker


def create_spec_worker(*args, **kwargs) -> "SpecPrefillWorker":
    spec_model_name = os.environ.get("spec_model", None)
    assert spec_model_name is not None

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

    def init_device(self) -> None:
        """Initialize device state, such as loading the model or other on-device
        memory allocations.
        """
        self.base_model_worker.init_device()
        self.base_model_worker.load_model()

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
        
        # make a copy for the spec model
        filtered_seq_group_metadata_list = []

        # we go over each request and check if it is a prompt data
        for seq_group_metadata in execute_model_req.seq_group_metadata_list:
            if seq_group_metadata.is_prompt:
                filtered_seq_group_metadata_list.append(seq_group_metadata)

        # we do spec prefill if we need to
        if len(filtered_seq_group_metadata_list) > 0:
            execute_model_req_clone = execute_model_req.clone(
                filtered_seq_group_metadata_list)

            spec_output = self.spec_model_worker.speculate_tokens(execute_model_req_clone)
            # TODO: drop indices

        return self.base_model_worker.execute_model(execute_model_req)

    def get_cache_block_size_bytes(self) -> int:
        raise NotImplementedError