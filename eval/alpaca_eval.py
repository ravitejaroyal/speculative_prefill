# https://github.com/tatsu-lab/alpaca_eval
from alpaca_eval.main import main

from models.llama.monkey_patch_llama import monkey_patch_llama

"""
CUDA_VISIBLE_DEVICES=0 VERBOSITY=2 KEEP=0.5 ALGO=attn python -m eval.alpaca_eval evaluate_from_model \
  --model_configs 'Meta-Llama-3.1-8B-Instruct' \
  --base_dir "./eval/alpaca_eval_models_configs" \
  --annotators_config 'alpaca_eval_gpt4_turbo_fn'
"""

if __name__ == "__main__":
    monkey_patch_llama()
    main()
