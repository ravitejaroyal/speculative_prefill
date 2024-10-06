import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig, LlamaConfig)

from models.speculative_prefill import build_speculative_prefill_model

input_text = """
Summarize the following text with a few words: 

As the Nameless officially do not exist, the upper echelons of the Gallian Army exploit 
the concept of plausible deniability in order to send them on missions that would otherwise 
make Gallia lose face in the war . While at times this works to their advantage , such as a 
successful incursion into Imperial territory , other orders cause certain members of the 422nd 
great distress . One such member , Gusurg , becomes so enraged that he abandons his post and defects 
into the ranks of Calamity Raven , attached to the ideal of Darcsen independence proposed by their 
leader , Dahau . At the same time , elements within Gallian Army Command move to erase the 
Nameless in order to protect their own interests . Hounded by both allies and enemies, 
and combined with the presence of a traitor within their ranks, 
the 422nd desperately move to keep themselves alive while at the same time 
fight to help the Gallian war effort. """


model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
messages = [{'role': 'user', 'content': input_text}]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer([prompt], return_tensors="pt")

input_ids = inputs['input_ids'].to('cuda')
attention_mask = inputs['attention_mask'].to('cuda')

# get keep indices
spec_prefill_model = build_speculative_prefill_model(keep_token_cnt=50)

spec_prefill_data = spec_prefill_model.speculative_prefill(
    input_ids=input_ids, 
    attention_mask=attention_mask
)

spec_prefill_inputs = spec_prefill_model.speculative_prefill_data_to_inputs(
    spec_prefill_data=spec_prefill_data, 
    input_ids=input_ids, 
    attention_mask=attention_mask
)

# messages = [{'role': 'user', 'content': input_text}]

# prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# inputs = tokenizer([prompt], return_tensors="pt")
# # inputs = tokenizer([input_text], return_tensors="pt")

# original_config_dict = LlamaConfig.from_pretrained(model_name).to_dict()

# new_config = LlamaTurboConfig.from_dict(
#     original_config_dict
# )

# new_config.use_mod_attn = True

# model = AutoModelForCausalLM.from_pretrained(
#     model_name, 
#     config=new_config, 
#     torch_dtype=torch.bfloat16, 
#     low_cpu_mem_usage=True, 
#     device_map="cuda", 
#     attn_implementation="flash_attention_2", 
#     trust_remote_code=True
# )

# # model = AutoModelForCausalLM.from_pretrained(
# #     model_name, 
# #     trust_remote_code=True, torch_dtype=torch.bfloat16, 
# #     device_map="auto", low_cpu_mem_usage=True, attn_implementation="flash_attention_2",
# # )


# throw_indices = torch.load("./tensors/throw_indices.pt").view(-1).cpu()
# keep_mask = torch.ones_like(inputs["input_ids"][0], dtype=torch.bool)
# keep_mask[throw_indices] = False
# position_ids = torch.arange(inputs["input_ids"].shape[-1])[None, ][:, keep_mask]
# input_ids = inputs["input_ids"][:, keep_mask]

# inputs = {
#     'input_ids': input_ids.to('cuda'), 
#     'attention_mask': torch.ones_like(input_ids).to('cuda'), 
#     'position_ids': position_ids.to('cuda')
# }
# # inputs = {'input_ids': inputs['input_ids'].to('cuda'), 'attention_mask': inputs['attention_mask'].to('cuda'),}

# gen_config = GenerationConfig(
#     do_sample=False, 
#     eos_token_id=128009, 
#     pad_token_id=128009
# )
# outputs = model.generate(**inputs, max_new_tokens=100, use_cache=False, return_dict_in_generate=True, output_scores=True, generation_config=gen_config) #use_cache=False, 

# print ("=====================")
# input_length = inputs['input_ids'].shape[1]
# generated_tokens = outputs.sequences[:, input_length:]
# print (generated_tokens.tolist())
# print (tokenizer.decode(generated_tokens.tolist()[0]))
