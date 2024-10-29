DEVICES=4,5,6,7

output_dir="./local/outputs/gsm8k"
mkdir -p $output_dir

CUDA_VISIBLE_DEVICES=$DEVICES lm_eval \
    --model vllm \
    --model_args pretrained=meta-llama/Meta-Llama-3.1-70B-Instruct,enforce_eager=True,max_model_len=16384,tensor_parallel_size=4,gpu_memory_utilization=0.9,enable_chunked_prefill=False \
    --tasks gsm8k_cot_llama_3.1_instruct \
    --apply_chat_template \
	--fewshot_as_multiturn \
    --num_fewshot 8 \
    --gen_kwargs temperature=0 \
    --batch_size 8 > $output_dir/baseline.txt

for exp in "p5_full" "p7_full" "p9_full"; do
    CUDA_VISIBLE_DEVICES=$DEVICES SPEC_CONFIG_PATH=./local/config_${exp}.yaml python -m eval.lm_eval \
        --model vllm \
        --model_args pretrained=meta-llama/Meta-Llama-3.1-70B-Instruct,enforce_eager=True,max_model_len=16384,tensor_parallel_size=4,gpu_memory_utilization=0.6,enable_chunked_prefill=False \
        --tasks gsm8k_cot_llama_3.1_instruct \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --num_fewshot 8 \
        --gen_kwargs temperature=0 \
        --batch_size 8 > $output_dir/${exp}_grad.txt
done
