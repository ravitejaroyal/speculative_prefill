DEVICES=4,5,6,7

mbpp_output_dir="./local/outputs/mbpp"
humaneval_output_dir="./local/outputs/humaneval"
mkdir -p $mbpp_output_dir
mkdir -p $humaneval_output_dir

CUDA_VISIBLE_DEVICES=$DEVICES evalplus.evaluate \
    --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --dataset humaneval \
    --backend vllm \
    --greedy \
    --enable-chunked-prefill False \
    --root ./local/evalplus_result/humaneval/baseline > $humaneval_output_dir/baseline.txt

CUDA_VISIBLE_DEVICES=$DEVICES evalplus.evaluate \
    --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --dataset mbpp \
    --backend vllm \
    --greedy \
    --enable_chunked_prefill False \
    --root ./local/evalplus_result/mbpp/baseline > $mbpp_output_dir/baseline.txt

for exp in "p5_32" "p5_full" "p7_32" "p7_full" "p9_32" "p9_full"; do
    CUDA_VISIBLE_DEVICES=$DEVICES SPEC_CONFIG_PATH=./local/config_${exp}.yaml python -m eval.evalplus \
        --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --dataset humaneval \
        --backend vllm \
        --greedy \
        --enable_chunked_prefill False \
        --root ./local/evalplus_result/humaneval/${exp} > $humaneval_output_dir/$exp.txt
    
    CUDA_VISIBLE_DEVICES=$DEVICES SPEC_CONFIG_PATH=./local/config_${exp}.yaml python -m eval.evalplus \
        --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --dataset mbpp \
        --backend vllm \
        --greedy \
        --enable_chunked_prefill False \
        --root ./local/evalplus_result/mbpp/${exp} > $mbpp_output_dir/$exp.txt
done

