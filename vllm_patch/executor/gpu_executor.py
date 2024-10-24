import os
from typing import Callable, Optional, Tuple, Type

from vllm.executor.gpu_executor import GPUExecutor
from vllm.worker.worker_base import WorkerBase


class PatchedGPUExecutor(GPUExecutor):
    def _get_worker_module_and_class(self) -> Tuple[str, str, Optional[Callable[[], Type[WorkerBase]]]]:
        """
            Patching this logic to consider using prefill speculator
        """
        worker_class_fn = None
        if os.environ.get("spec_model", None):
            worker_module_name = "vllm_patch.worker.spec_prefill_worker"
            worker_class_name = "create_spec_worker"
        elif self.scheduler_config.is_multi_step:
            worker_module_name = "vllm.worker.multi_step_worker"
            worker_class_name = "MultiStepWorker"
        elif self.speculative_config:
            worker_module_name = "vllm.spec_decode.spec_decode_worker"
            worker_class_name = "create_spec_worker"
        else:
            worker_module_name = "vllm.worker.worker"
            worker_class_name = "Worker"
        return (worker_module_name, worker_class_name, worker_class_fn)