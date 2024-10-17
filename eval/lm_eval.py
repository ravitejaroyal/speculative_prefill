from lm_eval.__main__ import cli_evaluate

from models.llama.monkey_patch_llama import monkey_patch_llama

if __name__ == "__main__":
    monkey_patch_llama()
    cli_evaluate()
