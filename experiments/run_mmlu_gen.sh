DEVICES=0,1,2,3

output_dir="./local/outputs/mmlu_generative"
mkdir -p $output_dir

CUDA_VISIBLE_DEVICES=$DEVICES lm_eval \
    --model vllm \
    --model_args pretrained=meta-llama/Meta-Llama-3.1-70B-Instruct,enforce_eager=True,max_model_len=16384,tensor_parallel_size=4,gpu_memory_utilization=0.9,enable_chunked_prefill=False \
    --tasks mmlu_generative \
    --gen_kwargs temperature=0,max_gen_toks=4 \
    --limit 256 \
    --batch_size 4 > $output_dir/baseline.txt

for exp in "p5_32" "p5_full" "p7_32" "p7_full" "p9_32" "p9_full"; do
    CUDA_VISIBLE_DEVICES=$DEVICES SPEC_CONFIG_PATH=./local/config_${exp}.yaml python -m eval.lm_eval \
        --model vllm \
        --model_args pretrained=meta-llama/Meta-Llama-3.1-70B-Instruct,enforce_eager=True,max_model_len=16384,tensor_parallel_size=4,gpu_memory_utilization=0.6,enable_chunked_prefill=False \
        --tasks mmlu_generative \
        --gen_kwargs temperature=0,max_gen_toks=4 \
        --limit 256 \
        --batch_size 4 > $output_dir/$exp.txt
done
