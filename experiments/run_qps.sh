OUTPUT_DIR="./local/outputs/qps"
NUM_SAMPLES=50
TIMEOUT=10
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
python3 eval/qps_client.py \
    --model $MODEL_NAME \
    --qps 0.2 \
    --category $CATEGORY \
    --max-tokens 1 \
    --timeout $TIMEOUT \
    --num-samples 4

echo "Start real profiling"
echo "" > $OUTPUT_DIR/${SIZE}_${3}_ttft_${CATEGORY}.txt

for qps in 0.1 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0; do
    echo "Sleep for 5 seconds"
    sleep 5
    python3 eval/qps_client.py \
        --model $MODEL_NAME \
        --qps $qps \
        --category $CATEGORY \
        --max-tokens 1 \
        --timeout $(( $TIMEOUT + 5 )) \
        --num-samples $NUM_SAMPLES >> $OUTPUT_DIR/${SIZE}_${3}_ttft_${CATEGORY}.txt
    
    if cat $OUTPUT_DIR/${SIZE}_${3}_ttft_${CATEGORY}.txt | grep -q "Found timeout in queries"; then
        exit 1
    fi
done
