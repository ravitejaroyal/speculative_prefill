from typing import Tuple

from vllm.sequence import ExecuteModelRequest
from vllm.worker.worker import Worker


class SpecWorker(Worker):
    """
        TODO: Disable KV behavior of a normal worker
    """

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        raise NotImplementedError

    def speculate_tokens(self, 
        execute_model_req: ExecuteModelRequest | None = None
    ):
        inputs = self.prepare_input(execute_model_req)
        if inputs is None:
            return None

        model_input, worker_input, kwargs = inputs
        num_steps = worker_input.num_steps

        self.execute_worker(worker_input)

        # If there is no input, we don't need to execute the model.
        if worker_input.num_seq_groups == 0:
            return []

        output = self.model_runner.execute_model(
            model_input=model_input,
            kv_caches=self.kv_cache[worker_input.virtual_engine]
            if self.kv_cache is not None else None,
            intermediate_tensors=None,
            num_steps=num_steps,
            **kwargs,
        )

        return None