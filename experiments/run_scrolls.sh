DEVICES=0,1,2,3

output_dir="./local/outputs/scrolls"
mkdir -p $output_dir

CUDA_VISIBLE_DEVICES=$DEVICES lm_eval \
    --model vllm \
    --model_args pretrained=meta-llama/Meta-Llama-3.1-70B-Instruct,enforce_eager=True,tensor_parallel_size=4 \
    --tasks scrolls_qasper_llama_16k,scrolls_govreport_llama_16k,scrolls_qasper_llama_16k \
    --gen_kwargs temperature=0 \
    --apply_chat_template \
    --limit 256 \
    --batch_size 1

for exp in "p3_32" "p3_full" "p5_32" "p5_full" "p7_32" "p7_full"; do
    CUDA_VISIBLE_DEVICES=$DEVICES SPEC_CONFIG_PATH=./local/config_${exp}.yaml python -m eval.lm_eval \
        --model vllm \
        --model_args pretrained=meta-llama/Meta-Llama-3.1-70B-Instruct,enforce_eager=True,max_model_len=32768,tensor_parallel_size=4,gpu_memory_utilization=0.6,enable_chunked_prefill=False \
        --tasks scrolls_narrativeqa_llama_16k,scrolls_qasper_llama_16k,scrolls_govreport_llama_16k \
        --gen_kwargs temperature=0 \
        --apply_chat_template \
        --limit 256 \
        --batch_size 1 > $output_dir/$exp.txt
done
