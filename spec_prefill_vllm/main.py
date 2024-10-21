from vllm_patch import enable_prefill_spec

# monkey patch must be placed before everything
enable_prefill_spec()

from vllm import LLM, SamplingParams
from vllm.executor.gpu_executor import GPUExecutor

print(GPUExecutor)

"""
    HUGGING_FACE_HUB_TOKEN=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 python main.py
"""

# llm = LLM(
#     'meta-llama/Llama-3.2-1B-Instruct', 
#     tokenizer='meta-llama/Meta-Llama-3.1-8B-Instruct'
# )
