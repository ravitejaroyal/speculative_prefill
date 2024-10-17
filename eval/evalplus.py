from evalplus.evaluate import main

from models.llama.monkey_patch_llama import monkey_patch_llama

"""
CUDA_VISIBLE_DEVICES=1 VERBOSITY=2 KEEP=0.7 ALGO=attn python -m eval.evalplus \
    --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --attn-implementation "flash_attention_2" \
    --dataset humaneval \
    --backend hf \
    --greedy \
    --root ./local/evalplus_results
"""

if __name__ == "__main__":
    monkey_patch_llama()
    main()