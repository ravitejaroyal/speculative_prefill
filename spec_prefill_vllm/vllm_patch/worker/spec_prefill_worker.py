import os
from typing import List, Tuple

import torch
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.worker.model_runner import ModelRunner
from vllm.worker.worker import Worker
from vllm.worker.worker_base import LoraNotSupportedWorkerBase
from vllm_patch.worker.spec_worker_base import SpecWorker


def create_spec_worker(*args, **kwargs) -> "SpecPrefillWorker":
    spec_model_name = os.environ.get("spec_model", None)
    assert spec_model_name is not None

    # create the base model
    kwargs["model_runner_cls"] = ModelRunner
    base_model_worker = Worker(*args, **kwargs) 

    # create the spec prefill model (assume the same config as before)
    spec_model_kwargs = kwargs.copy()
    kwargs["model_config"].model = spec_model_name
    spec_model_worker = SpecWorker(*args, **spec_model_kwargs)

    return SpecPrefillWorker(
        base_model_worker=base_model_worker, 
        spec_model_worker=spec_model_worker
    )


class SpecPrefillWorker(LoraNotSupportedWorkerBase):
    def __init__(
        self, 
        base_model_worker: Worker, 
        spec_model_worker: Worker
    ):
        self.base_model_worker = base_model_worker
        self.spec_model_worker = spec_model_worker

    def init_device(self) -> None:
        """Initialize device state, such as loading the model or other on-device
        memory allocations.
        """
        self.base_model_worker.init_device()
        self.spec_model_worker.init_device()

        self.base_model_worker.load_model()
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
        self.spec_model_worker.initialize_cache(
            num_gpu_blocks=num_gpu_blocks, 
            num_cpu_blocks=num_cpu_blocks
        )
    
    @torch.inference_mode
    def execute_model(
        self, 
        execute_model_req: ExecuteModelRequest | None = None
    ) -> List[SamplerOutput] | None:
        pass

    @torch.inference_mode
    def start_worker_execution_loop(self) -> None:
        pass

    def get_cache_block_size_bytes(self) -> int:
        raise NotImplementedError