import random
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

spec_model_name = 'meta-llama/Llama-3.2-3B-Instruct'
base_model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
tokenizer_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'

seq_len = 512
num_samples = 32
seed = 227

# seed everything
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

# load data
dataset = load_dataset(
    "allenai/paloma", 
    name="c4_en", 
    split="test"
)
texts = dataset.shuffle(seed=seed)["text"]

# tokenizer data into batch
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token

# we want to find examples whose tokenized len >= seqlen
samples = []
for text in texts:
    tokenized_output = tokenizer(
        text, 
        truncation=True, 
        max_length=seq_len, 
        return_tensors='pt'
    )
    input_ids = tokenized_output["input_ids"][0]
    if len(input_ids) >= seq_len:
        samples.append(input_ids)
    if len(samples) == num_samples:
        break

print(f"Found {len(samples)} samples with number of tokens >= {seq_len}")
samples = torch.stack(samples, dim=0).to('cuda')

# get attn scores
@torch.inference_mode
def get_attn_scores(model_name) -> Tuple[torch.Tensor]:
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        attn_implementation="eager", # for outputing attn
    )

    output = model.forward(
        input_ids=samples, 
        output_attentions=True, 
        return_dict=True
    )

    # layer x [bs, head, q_len, k_len]
    return output.attentions

# store attn scores
spec_attns = get_attn_scores(spec_model_name)
base_attns = get_attn_scores(base_model_name)

# visualize relationship between two models
correlation = torch.zeros((len(spec_attns), len(base_attns)))

for si in range(len(spec_attns)):
    for bi in range(len(base_attns)):
        # [bs, head, q_len, k_len]
        # max over heads
        sa = torch.max(spec_attns[si][..., -1, :], dim=1)[0].to(torch.float32)
        ba = torch.max(base_attns[bi][..., -1, :], dim=1)[0].to(torch.float32)

        sa = torch.nn.functional.log_softmax(sa, dim=-1)
        ba = torch.nn.functional.softmax(ba, dim=-1)

        score = torch.nn.functional.kl_div(sa, ba, reduction='batchmean')

        correlation[si, bi] = score

sns.heatmap(correlation, square=True)
plt.xlabel("Layer idx of base model")
plt.ylabel("Layer idx of spec model")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('./visualization/kl_3b8b.png', dpi=300)
