output_dir="./local/outputs/mmlu_generative"
mkdir -p $output_dir

{
    CUDA_VISIBLE_DEVICES=0 lm_eval \
        --model vllm \
        --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,enforce_eager=True,max_model_len=16384,gpu_memory_utilization=0.6,enable_chunked_prefill=False \
        --tasks mmlu_generative \
        --gen_kwargs temperature=0,max_gen_toks=4 \
        --limit 256 \
        --batch_size 1 > $output_dir/baseline_8b.txt
} &

{
    CUDA_VISIBLE_DEVICES=1 SPEC_CONFIG_PATH=./local/config_attn_p5.yaml python -m eval.lm_eval \
        --model vllm \
        --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,enforce_eager=True,max_model_len=16384,gpu_memory_utilization=0.6,enable_chunked_prefill=False \
        --tasks mmlu_generative \
        --gen_kwargs temperature=0,max_gen_toks=4 \
        --limit 256 \
        --batch_size 1 > $output_dir/attn_p5_8b.txt
} &

{
    CUDA_VISIBLE_DEVICES=2 SPEC_CONFIG_PATH=./local/config_attn_p9.yaml python -m eval.lm_eval \
        --model vllm \
        --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,enforce_eager=True,max_model_len=16384,gpu_memory_utilization=0.6,enable_chunked_prefill=False \
        --tasks mmlu_generative \
        --gen_kwargs temperature=0,max_gen_toks=4 \
        --limit 256 \
        --batch_size 1 > $output_dir/attn_p9_8b.txt
} &

{
    CUDA_VISIBLE_DEVICES=3 SPEC_CONFIG_PATH=./local/config_attn_lah_2_p5.yaml python -m eval.lm_eval \
        --model vllm \
        --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,enforce_eager=True,max_model_len=16384,gpu_memory_utilization=0.6,enable_chunked_prefill=False \
        --tasks mmlu_generative \
        --gen_kwargs temperature=0,max_gen_toks=4 \
        --limit 256 \
        --batch_size 1 > $output_dir/attn_lah_2_p5_8b.txt
} &

{
    CUDA_VISIBLE_DEVICES=4 SPEC_CONFIG_PATH=./local/config_attn_lah_4_p5.yaml python -m eval.lm_eval \
        --model vllm \
        --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,enforce_eager=True,max_model_len=16384,gpu_memory_utilization=0.6,enable_chunked_prefill=False \
        --tasks mmlu_generative \
        --gen_kwargs temperature=0,max_gen_toks=4 \
        --limit 256 \
        --batch_size 1 > $output_dir/attn_lah_4_p5_8b.txt
} &

{
    CUDA_VISIBLE_DEVICES=5 SPEC_CONFIG_PATH=./local/config_attn_lah_2_p9.yaml python -m eval.lm_eval \
        --model vllm \
        --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,enforce_eager=True,max_model_len=16384,gpu_memory_utilization=0.6,enable_chunked_prefill=False \
        --tasks mmlu_generative \
        --gen_kwargs temperature=0,max_gen_toks=4 \
        --limit 256 \
        --batch_size 1 > $output_dir/attn_lah_2_p9_8b.txt
} &

{
    CUDA_VISIBLE_DEVICES=6 SPEC_CONFIG_PATH=./local/config_attn_lah_4_p9.yaml python -m eval.lm_eval \
        --model vllm \
        --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,enforce_eager=True,max_model_len=16384,gpu_memory_utilization=0.6,enable_chunked_prefill=False \
        --tasks mmlu_generative \
        --gen_kwargs temperature=0,max_gen_toks=4 \
        --limit 256 \
        --batch_size 1 > $output_dir/attn_lah_4_p9_8b.txt
} &
