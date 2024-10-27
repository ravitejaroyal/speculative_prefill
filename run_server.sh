python -m speculative_prefill.scripts serve \
    'meta-llama/Meta-Llama-3.1-8B-Instruct' \
    --dtype auto \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.8 \
    --enable-chunked-prefill=False \
    --enforce-eager True \
    --api-key local_server
