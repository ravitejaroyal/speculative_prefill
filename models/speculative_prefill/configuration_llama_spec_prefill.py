from transformers.models.llama.configuration_llama import LlamaConfig


class LlamaSpecPrefillConfig(LlamaConfig):
    model_type = "llama_spec_prefill"
    keys_to_ignore_at_inference = ["past_key_values"]
    
    def __init__(
        self, 
        keep_token_cnt: int = -1, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.keep_token_cnt = keep_token_cnt
