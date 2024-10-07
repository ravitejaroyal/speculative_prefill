"""
    A prefill speculator will:
    1. first get the full context, predict the thrown away indices based on the
        first layer
    2. send the indices to the main model
    3. finish the full prefill and send the KV cache to the main model
"""

import torch
from transformers import (AutoConfig, AutoModel, AutoTokenizer, LlamaConfig,
                          LlamaTokenizer)
from transformers.models.auto.modeling_auto import MODEL_MAPPING

from .configuration_llama_spec_prefill import LlamaSpecPrefillConfig
from .modeling_llama_spec_prefill import LlamaSpecPrefillModel

AutoConfig.register("llama_spec_prefill", LlamaSpecPrefillConfig)
AutoTokenizer.register(LlamaSpecPrefillConfig, LlamaTokenizer)
AutoModel.register(LlamaSpecPrefillConfig, LlamaSpecPrefillModel)
MODEL_MAPPING["llama_spec_prefill"] = "LlamaSpecPrefillModel"


def build_speculative_prefill_model(**kwargs) -> LlamaSpecPrefillModel:
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    original_config_dict = LlamaConfig.from_pretrained(model_name).to_dict()
    original_config_dict.update(**kwargs)
    spec_prefill_model_config = LlamaSpecPrefillConfig.from_dict(
        original_config_dict
    )

    return AutoModel.from_pretrained(
        model_name, 
        config=spec_prefill_model_config, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        device_map="cuda", 
        attn_implementation="flash_attention_2", 
        trust_remote_code=True
    )
