from lm_eval.__main__ import cli_evaluate

from models.llama.monkey_patch_llama import monkey_patch_llama

"""
    lm_eval \
        --model hf \
        --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,attn_implementation="flash_attention_2" \
        --task mmlu_generative \
        --batch_size auto \
        --device cuda:1 \
        --limit 50 \
        --apply_chat_template \
        --gen_kwargs "do_sample=False,eos_token_id=128009"

    python eval.py \
        --model hf \
        --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,attn_implementation="flash_attention_2" \
        --task mmlu_generative \
        --batch_size 1 \
        --device cuda:0 \
        --limit 50 \
        --apply_chat_template \
        --gen_kwargs "do_sample=False,eos_token_id=128009"
"""

if __name__ == "__main__":
    monkey_patch_llama()
    cli_evaluate()
