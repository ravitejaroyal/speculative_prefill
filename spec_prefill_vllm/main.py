from vllm_patch import enable_prefill_spec

# monkey patch must be placed before everything
enable_prefill_spec(spec_model='meta-llama/Llama-3.2-1B-Instruct')

from vllm import LLM, SamplingParams
from vllm.executor.gpu_executor import GPUExecutor

print(GPUExecutor)

"""
    HUGGING_FACE_HUB_TOKEN=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 python main.py
"""

llm = LLM(
    'meta-llama/Llama-3.2-1B-Instruct', 
    tokenizer='meta-llama/Meta-Llama-3.1-8B-Instruct'
)

# conversation = [
#     {
#         "role": "system",
#         "content": "You are a helpful assistant"
#     },
#     {
#         "role": "user",
#         "content": "Hello"
#     },
#     {
#         "role": "assistant",
#         "content": "Hello! How can I assist you today?"
#     },
#     {
#         "role": "user",
#         "content": "Write an essay about the importance of higher education.",
#     },
# ]

# print(llm.chat(conversation))

print(llm.generate(
    [
        "Where is the Beijing City? ", 
        "Tell me a bit about the importance of higher education. "
    ], 
    SamplingParams(
        max_tokens=2, 
        temperature=0.0
    )
))