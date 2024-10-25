from speculative_prefill import enable_prefill_spec

enable_prefill_spec(
    spec_model='meta-llama/Llama-3.2-1B-Instruct', 
    spec_config_path='./local/config.yaml'
)

from vllm.scripts import main

if __name__ == "__main__":    
    main()
