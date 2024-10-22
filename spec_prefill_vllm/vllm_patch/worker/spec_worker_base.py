from typing import List, Optional, Tuple

import torch
from vllm.worker.worker import Worker


class SpecWorker(Worker):
    """
        Disable KV behavior of a normal worker
    """

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        raise NotImplementedError

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        pass

    def get_cache_block_size_bytes(self) -> int:
        return 0
    
    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return self.gpu_cache