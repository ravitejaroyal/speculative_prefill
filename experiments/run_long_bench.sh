cd ./eval/long_bench

{
    CUDA_VISIBLE_DEVICES=0 python pred_vllm.py \
        --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --exp baseline_again \
        --e
} &

wait

# run in parallel
{
    CUDA_VISIBLE_DEVICES=0,1 SPEC_CONFIG_PATH=../../local/config_grad_p3.yaml python pred_vllm.py \
        --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --e \
        --spec-prefill \
        --gpu-memory-utilization 0.6 \
        --tensor-parallel-size 2 \
        --no-8k \
        --exp grad_p3

    python eval.py --e --exp grad_p3
} &

{
    CUDA_VISIBLE_DEVICES=2,3 SPEC_CONFIG_PATH=../../local/config_grad_p5.yaml python pred_vllm.py \
        --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --e \
        --spec-prefill \
        --gpu-memory-utilization 0.6 \
        --tensor-parallel-size 2 \
        --no-8k \
        --exp grad_p5

    python eval.py --e --exp grad_p5
} &

{
    CUDA_VISIBLE_DEVICES=4,5 SPEC_CONFIG_PATH=../../local/config_grad_p7.yaml python pred_vllm.py \
        --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --e \
        --spec-prefill \
        --gpu-memory-utilization 0.6 \
        --tensor-parallel-size 2 \
        --no-8k \
        --exp grad_p7

    python eval.py --e --exp grad_p7
} &

{
    CUDA_VISIBLE_DEVICES=6,7 SPEC_CONFIG_PATH=../../local/config_grad_p9.yaml python pred_vllm.py \
        --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --e \
        --spec-prefill \
        --gpu-memory-utilization 0.6 \
        --tensor-parallel-size 2 \
        --no-8k \
        --exp grad_p9

    python eval.py --e --exp grad_p9
} &

wait

{
    CUDA_VISIBLE_DEVICES=0,1 SPEC_CONFIG_PATH=../../local/config_attn_p3.yaml python pred_vllm.py \
        --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --e \
        --spec-prefill \
        --gpu-memory-utilization 0.6 \
        --tensor-parallel-size 2 \
        --exp attn_p3

    python eval.py --e --exp attn_p3
} &

{
    CUDA_VISIBLE_DEVICES=2,3 SPEC_CONFIG_PATH=../../local/config_attn_p5.yaml python pred_vllm.py \
        --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --e \
        --spec-prefill \
        --gpu-memory-utilization 0.6 \
        --tensor-parallel-size 2 \
        --exp attn_p5

    python eval.py --e --exp attn_p5
} &

{
    CUDA_VISIBLE_DEVICES=4,5 SPEC_CONFIG_PATH=../../local/config_attn_p7.yaml python pred_vllm.py \
        --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --e \
        --spec-prefill \
        --gpu-memory-utilization 0.6 \
        --tensor-parallel-size 2 \
        --exp attn_p7

    python eval.py --e --exp attn_p7
} &

{
    CUDA_VISIBLE_DEVICES=6,7 SPEC_CONFIG_PATH=../../local/config_attn_p9.yaml python pred_vllm.py \
        --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --e \
        --spec-prefill \
        --gpu-memory-utilization 0.6 \
        --tensor-parallel-size 2 \
        --exp attn_p9

    python eval.py --e --exp attn_p9
} &
