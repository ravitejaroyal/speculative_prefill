cd ./eval/long_bench

python pred_vllm.py \
    --model "/data_persistent3/jingyu/models--neuralmagic--Meta-Llama-3.1-405B-Instruct-FP8/snapshots/ca6a4018496537951173ffa7345083883edf69ff" \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.95 \
    --exp baseline_405b

python eval.py --exp baseline_405b

for exp in "p1_full_lah8" "p3_full_lah8" "p5_full_lah8" "p7_full_lah8" "p9_full_lah8"; do
    SPEC_CONFIG_PATH=../../configs/config_${exp}.yaml python pred_vllm.py \
        --model "/data_persistent3/jingyu/models--neuralmagic--Meta-Llama-3.1-405B-Instruct-FP8/snapshots/ca6a4018496537951173ffa7345083883edf69ff" \
        --spec-model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --spec-prefill \
        --tensor-parallel-size 8 \
        --gpu-memory-utilization 0.95 \
        --exp spec_${exp}

    python eval.py --exp spec_${exp}_405b_8b
done