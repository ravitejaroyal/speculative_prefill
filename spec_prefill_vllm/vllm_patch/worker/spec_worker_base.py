from abc import abstractmethod

import torch
from transformers import AutoModelForCausalLM
from vllm.sequence import ExecuteModelRequest


class SpecWorker:
    @abstractmethod
    def speculate_tokens(
        self, 
        execute_model_req: ExecuteModelRequest | None = None
    ):
        raise NotImplementedError


class HFSpecWorker(SpecWorker):
    def __init__(
        self, 
        spec_model_name: str
    ) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            spec_model_name, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            attn_implementation="flash_attention_2",
        )

    def speculate_tokens(
        self, 
        execute_model_req: ExecuteModelRequest | None = None
    ):
        hf_inputs = self._extract_hf_inputs(execute_model_req)
        pass
    
    def _extract_hf_inputs(
        self, 
        execute_model_req: ExecuteModelRequest | None = None
    ):
        pass