from vllm_patch import enable_prefill_spec

enable_prefill_spec(
    spec_model='meta-llama/Llama-3.2-1B-Instruct', 
    spec_config_path='./local/config.yaml'
)

from evalplus.evaluate import main

"""
python -m eval.evalplus \
    --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --dataset humaneval \
    --backend vllm \
    --greedy \
    --root ./local/evalplus_result
"""

if __name__ == "__main__":
    main()
