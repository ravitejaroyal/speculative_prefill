OUTPUT_DIR="./local/outputs/qps"
TIMEOUT=20
NUM_SAMPLES=64
mkdir -p $OUTPUT_DIR

SIZE=${1:-"70B"}
CATEGORY=summarization

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
    --max-tokens 8 \
    --timeout $(( $TIMEOUT + 5 )) \
    --num-samples 4

echo "Start real profiling"
echo "" > $OUTPUT_DIR/${SIZE}_${2}_ttft_${CATEGORY}_hqps.txt

for qps in 3.2 3.4 3.6 3.8 4.0 4.2 4.4 4.6 4.8 5.0 5.2 5.4 5.6 5.8 6.0 6.2 6.4 6.6 6.8 7.0 7.2 7.4 7.6 7.8 8.0; do
    echo "Sleep for 10 seconds"
    sleep 10
    python3 eval/qps_client.py \
        --model $MODEL_NAME \
        --qps $qps \
        --category $CATEGORY \
        --timeout $(( $TIMEOUT + 5 )) \
        --num-samples $NUM_SAMPLES >> $OUTPUT_DIR/${SIZE}_${2}_ttft_${CATEGORY}_hqps.txt

    if cat $OUTPUT_DIR/${SIZE}_${2}_ttft_${CATEGORY}_hqps.txt | grep -q "Found timeout in queries"; then
        exit 1
    fi
done
