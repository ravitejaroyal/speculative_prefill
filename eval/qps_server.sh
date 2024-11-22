SIZE=${2:-"70B"}
API_KEY=${3:-local_server}
PORT=${4:-8888}

SPEC_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
MODEL_NAME=meta-llama/Meta-Llama-3.1-${SIZE}-Instruct

echo "Using port ${PORT} and api key ${API_KEY}"

fuser -n tcp ${PORT}

if [ "$1" = "native" ]; then
    echo "Starting native vllm serving"
    python -m speculative_prefill.scripts serve \
        ${MODEL_NAME} \
        --dtype auto \
        --max-model-len 65536 \
        --gpu-memory-utilization 0.9 \
        --enable-chunked-prefill=False \
        --tensor-parallel-size 8 \
        --enforce-eager \
        --max-num-seqs 128 \
        --api-key=$API_KEY \
        --port=$PORT

elif [ "$1" = "spec_prefill" ]; then
    echo "Starting vllm serving with speculative prefill" 
    SPEC_CONFIG_PATH=./configs/config_p1_full_lah8.yaml ENABLE_SP=$SPEC_NAME python -m speculative_prefill.scripts serve \
        ${MODEL_NAME} \
        --dtype auto \
        --max-model-len 65536 \
        --gpu-memory-utilization 0.9 \
        --enable-chunked-prefill=False \
        --tensor-parallel-size 8 \
        --enforce-eager \
        --max-num-seqs 128 \
        --api-key=$API_KEY \
        --port=$PORT
else
    echo "Invalid serving type..."
fi