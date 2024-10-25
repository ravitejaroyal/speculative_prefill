from speculative_prefill import enable_prefill_spec

enable_prefill_spec(
    spec_model='meta-llama/Llama-3.2-1B-Instruct', 
    spec_config_path='./local/config.yaml'
)

from vllm.scripts import main

"""
    python -m speculative_prefill.scripts serve \
        'meta-llama/Meta-Llama-3.1-8B-Instruct' \
        --dtype auto \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.8 \
        --enable-chunked-prefill=False
"""

if __name__ == "__main__":    
    main()
