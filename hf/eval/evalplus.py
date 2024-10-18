from typing import List

from evalplus.evaluate import main
from evalplus.provider.hf import HuggingFaceDecoder
from evalplus.provider.utility import make_raw_chat_prompt
from stop_sequencer import StopSequencer

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

def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1

        prompt = (
            prompt
            if self.is_direct_completion()
            else make_raw_chat_prompt(
                prompt, self.instruction_prefix, self.response_prefix, self.tokenizer
            )
        )
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.device
        )
        kwargs = {}
        if do_sample:
            kwargs["top_p"] = 0.95
            kwargs["temperature"] = self.temperature

        stop_sequencer = StopSequencer(
            self.model,
            model_type="causal",  # or seq2seq
            tokenizer=self.tokenizer,
        )

        model = stop_sequencer.register_stop_texts(
            stop_texts=self.eos,
            input_length=input_tokens.size(-1),
        )

        outputs = model.generate(
            input_tokens,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            num_return_sequences=min(self.batch_size, num_samples),
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        gen_strs = self.tokenizer.batch_decode(
            outputs[:, input_tokens.size(-1) :],
            skip_special_tokens=self.skip_special_tokens,
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index].replace("\t", "    "))
        return outputs


if __name__ == "__main__":
    monkey_patch_llama()
    HuggingFaceDecoder.codegen = codegen
    main()