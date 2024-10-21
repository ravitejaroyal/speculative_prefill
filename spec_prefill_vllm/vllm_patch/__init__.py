import os
from typing import Optional

from vllm_patch.executor import patch_executor
from vllm_patch.worker import patch_worker


def enable_prefill_spec(
    spec_model: str = 'meta-llama/Llama-3.2-1B-Instruct'
):
    print("Setting up environment vars...")
    os.environ["spec_model"] = spec_model

    print("Applying speculative prefill vllm monkey patch...")
    patch_executor()
    patch_worker()
