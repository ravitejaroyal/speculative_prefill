cd ./eval/long_bench

for exp in "pp1" "p1" "pp1_full" "p1_full" "pp1_full_lah8" "p1_full_lah8"; do
    SPEC_CONFIG_PATH=../../configs/config_${exp}.yaml python pred_vllm.py \
        --model "meta-llama/Meta-Llama-3.1-70B-Instruct" \
        --spec-model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --spec-prefill \
        --tensor-parallel-size 8 \
        --exp spec_${exp}

    python eval.py --exp spec_${exp}
done
