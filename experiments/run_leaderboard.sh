DEVICES=0,1,2,3

output_dir="./local/outputs/leaderboard_ifeval"
mkdir -p $output_dir

# CUDA_VISIBLE_DEVICES=$DEVICES lm_eval \
#     --model vllm \
#     --model_args pretrained=meta-llama/Meta-Llama-3.1-70B-Instruct,enforce_eager=True,max_model_len=16384,tensor_parallel_size=4,gpu_memory_utilization=0.9,enable_chunked_prefill=False \
#     --tasks leaderboard_ifeval \
#     --apply_chat_template \
#     --gen_kwargs temperature=0 \
#     --batch_size 8 > $output_dir/baseline.txt

for exp in "attn_p3" "attn_p5" "attn_p7"; do
    CUDA_VISIBLE_DEVICES=$DEVICES SPEC_CONFIG_PATH=./local/config_${exp}.yaml python -m eval.lm_eval \
        --model vllm \
        --model_args pretrained=meta-llama/Meta-Llama-3.1-70B-Instruct,enforce_eager=True,max_model_len=16384,tensor_parallel_size=4,gpu_memory_utilization=0.6,enable_chunked_prefill=False \
        --tasks leaderboard_ifeval \
        --apply_chat_template \
        --gen_kwargs temperature=0 \
        --batch_size 8 > $output_dir/${exp}_attn.txt
done
