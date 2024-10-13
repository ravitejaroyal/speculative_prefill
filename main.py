import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from models.llama.monkey_patch_llama import monkey_patch_llama

# input_text = """
# What is the last word in the below text: 

# Sven Magnus Øen Carlsen[a] (born 30 November 1990) is a Norwegian chess grandmaster. Carlsen is a five-time World Chess Champion, the reigning five-time World Rapid Chess Champion, the reigning seven-time World Blitz Chess Champion, and the reigning Chess World Cup Champion. He has held the No. 1 position in the FIDE world chess rankings since 1 July 2011 and trails only Garry Kasparov in time spent as the highest-rated player in the world.[1] His peak rating of 2882 is the highest in history. He also holds the record for the longest unbeaten streak at an elite level in classical chess at 125 games.[2][3]

# A chess prodigy, Carlsen finished first in the C group of the Corus chess tournament shortly after he turned 13 and earned the title of grandmaster a few months later. At 15, he won the Norwegian Chess Championship, and later became the youngest ever player to qualify for the Candidates Tournament in 2005.[1] At 17, he finished joint first in the top group of Corus. He surpassed a rating of 2800 at 18, the youngest at the time to do so. In 2010, at 19, he reached No. 1 in the FIDE world rankings, the youngest person ever to do so.

# Carlsen became World Chess Champion in 2013 by defeating Viswanathan Anand. He retained his title against Anand the following year and won both the 2014 World Rapid Championship and World Blitz Championship, becoming the first player to hold all three titles simultaneously, a feat which he repeated in 2019 and 2022.[4][5] He defended his classical world title against Sergey Karjakin in 2016, Fabiano Caruana in 2018, and Ian Nepomniachtchi in 2021. Carlsen declined to defend his title in 2023, citing a lack of motivation.[6]
# """

input_text = ""
for i in range(1500):
    input_text += f"line {i}, key: 123456\n"
input_text += "line 1500, key: 245678\n"
for i in range(300):
    input_text += f"line {1501 + i}, key: 123456\n"
input_text += "\nWhich line is the key 245678?"

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
messages = [{'role': 'user', 'content': input_text}]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# prompt = input_text
inputs = tokenizer([prompt], return_tensors="pt")

input_ids = inputs['input_ids'].to('cuda')
attention_mask = inputs['attention_mask'].to('cuda')

print(f"Context length: {input_ids.shape[-1]}")

monkey_patch_llama()

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16, 
    device_map="auto", 
    low_cpu_mem_usage=True, 
    attn_implementation="flash_attention_2",
)

gen_config = GenerationConfig(
    do_sample=False, 
    eos_token_id=128009, 
    pad_token_id=128009
)

outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,  
    max_new_tokens=50, 
    use_cache=True, 
    return_dict_in_generate=True, 
    output_scores=True, 
    generation_config=gen_config
)

print("=====================")
input_length = inputs["input_ids"].shape[1]
generated_tokens = outputs.sequences[:, input_length:]
print(generated_tokens.tolist())
print(tokenizer.decode(generated_tokens.tolist()[0]))
