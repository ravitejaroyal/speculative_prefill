NUM_SAMPLES=64
OUTPUT_DIR="./local/outputs/qps"

mkdir -p $OUTPUT_DIR

SIZE=${1:-"70B"}
CATEGORY=${2:-"few-shot-learning"}

if [ $SIZE == "70B" ]; then
    MODEL_NAME=meta-llama/Meta-Llama-3.1-70B-Instruct
elif [ $SIZE == "405B" ]; then
    MODEL_NAME=neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8
else
    echo "Invalid model size"
    exit 1
fi

# warmup
echo "Warm up server"
python eval/qps_client.py \
    --model $MODEL_NAME \
    --qps 0.5 \
    --category $CATEGORY \
    --max-tokens 1 \
    --timeout 30 \
    --num-samples 2

echo "Start real profiling"
for qps in 1 2 4 6 8 10 12 14 16; do
    echo "Sleep for 5 seconds"
    sleep 5
    python eval/qps_client.py \
        --model $MODEL_NAME \
        --qps $qps \
        --category $CATEGORY \
        --max-tokens 1 \
        --timeout 30 \
        --num-samples $NUM_SAMPLES >> $OUTPUT_DIR/${SIZE}_${3}_ttft_${CATEGORY}.txt
done
