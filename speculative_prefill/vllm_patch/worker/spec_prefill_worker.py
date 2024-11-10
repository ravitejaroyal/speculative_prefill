import os
from typing import List, Tuple

import torch
from vllm.config import ModelConfig
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.worker.model_runner import ModelRunner
from vllm.worker.worker import Worker
from vllm.worker.worker_base import LoraNotSupportedWorkerBase, WorkerBase


def split_num_cache_blocks_evenly(
    base_model_cache_block_size_bytes: int,
    spec_model_cache_block_size_bytes: int,
    total_num_gpu_blocks: int
) -> int:
    new_num_gpu_blocks = int(
        total_num_gpu_blocks * base_model_cache_block_size_bytes /
        (spec_model_cache_block_size_bytes + base_model_cache_block_size_bytes))

    return new_num_gpu_blocks


def create_spec_worker(*args, **kwargs) -> "SpecPrefillWorker":
    spec_model_name = os.environ.get("SPEC_MODEL", None)
    assert spec_model_name is not None

    assert kwargs["scheduler_config"].chunked_prefill_enabled == False, \
        "Please set --enable-chunked-prefill=False or enable_chunked_prefill=False. "

    # create the base model
    kwargs["model_runner_cls"] = ModelRunner
    model_config: ModelConfig = kwargs['model_config']
    base_model_worker = Worker(*args, **kwargs) 

    # create the spec prefill model
    # TODO: deal with this later in a proper way
    spec_kwargs = kwargs.copy()
    spec_kwargs["model_config"] = ModelConfig(
        model=spec_model_name, 
        tokenizer=model_config.tokenizer,
        tokenizer_mode=model_config.tokenizer_mode,
        trust_remote_code=model_config.trust_remote_code,
        dtype=model_config.dtype,
        seed=model_config.seed,
        revision=model_config.revision,
        code_revision=model_config.code_revision,
        rope_scaling=model_config.rope_scaling,
        rope_theta=model_config.rope_theta,
        tokenizer_revision=model_config.tokenizer_revision,
        max_model_len=model_config.max_model_len,
        quantization=model_config.quantization,
        quantization_param_path=model_config.quantization_param_path,
        enforce_eager=model_config.enforce_eager,
        max_seq_len_to_capture=model_config.max_seq_len_to_capture,
        max_logprobs=model_config.max_logprobs,
        disable_sliding_window=model_config.disable_sliding_window,
        skip_tokenizer_init=model_config.skip_tokenizer_init,
        served_model_name=model_config.served_model_name,
        use_async_output_proc=model_config.use_async_output_proc,
        override_neuron_config=model_config.override_neuron_config
    )
    spec_model_worker = Worker(*args, **spec_kwargs)

    return SpecPrefillWorker(
        base_model_worker=base_model_worker, 
        spec_model_worker=spec_model_worker
    )


class SpecPrefillWorker(LoraNotSupportedWorkerBase):
    def __init__(
        self,
        base_model_worker: WorkerBase, 
        spec_model_worker: WorkerBase, 
    ):
        self.base_model_worker = base_model_worker
        self.spec_model_worker = spec_model_worker
    
    def init_device(self) -> None:
        # The base worker model is initialized first in case the spec
        # model has a smaller TP degree than the base worker.
        self.base_model_worker.init_device()
        self.spec_model_worker.init_device()

        self.base_model_worker.load_model()
        self.spec_model_worker.load_model()

    def load_model(self, *args, **kwargs):
        pass

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        num_gpu_blocks, num_cpu_blocks = (
            self.base_model_worker.determine_num_available_blocks())

        base_model_cache_block_size_bytes = (
            self.base_model_worker.get_cache_block_size_bytes())
        spec_model_cache_block_size_bytes = (
            self.spec_model_worker.get_cache_block_size_bytes())

        new_num_gpu_blocks = split_num_cache_blocks_evenly(
            base_model_cache_block_size_bytes, 
            spec_model_cache_block_size_bytes, 
            num_gpu_blocks
        )
        
        return new_num_gpu_blocks, num_cpu_blocks

    def initialize_cache(
        self, num_gpu_blocks: int,
        num_cpu_blocks: int
    ) -> None:
        self.base_model_worker.initialize_cache(
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks)
        self.spec_model_worker.initialize_cache(
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks)
    
    @torch.inference_mode
    def execute_model(
        self, 
        execute_model_req: ExecuteModelRequest | None = None
    ) -> List[SamplerOutput] | None:
        return self.base_model_worker.execute_model(execute_model_req)

    @torch.inference_mode()
    def start_worker_execution_loop(self) -> None:
        """Execute model loop to perform speculative decoding
        in parallel worker."""
        while self._run_non_driver_rank():
            pass

    def get_cache_block_size_bytes(self) -> int:
        raise NotImplementedError