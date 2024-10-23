from vllm.sequence import ExecuteModelRequest


class SpecWorker:
    def __init__(
        self, 
        spec_model_name: str
    ) -> None:
        pass

    def speculate_tokens(
        self, 
        execute_model_req: ExecuteModelRequest | None = None
    ):
        return None