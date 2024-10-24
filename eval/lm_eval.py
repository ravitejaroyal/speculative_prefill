from vllm_patch import enable_prefill_spec

enable_prefill_spec(
    spec_model='meta-llama/Llama-3.2-1B-Instruct', 
    spec_config_path='./local/config.yaml'
)

from lm_eval.__main__ import cli_evaluate

"""
python -m eval.lm_eval \
    --model vllm \
    --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,dtype=auto,gpu_memory_utilization=0.8,enable_chunked_prefill=False \
    --tasks mmlu_generative \
    --gen_kwargs do_sample=False,max_gen_toks=2 \
    --limit 100 \
    --batch_size 2
"""

if __name__ == "__main__":
    cli_evaluate()
