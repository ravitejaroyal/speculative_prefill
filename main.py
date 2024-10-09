import torch
from networkx import general_random_intersection_graph
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from models.llama.monkey_patch_llama import monkey_patch_llama

input_text = """
Summarize the following text with a one sentence: 

Upon the death of Lord Jon Arryn, the principal advisor to King Robert Baratheon, Robert recruits his childhood friend Eddard "Ned" Stark, now Warden of the North, to replace Arryn as Hand of the King, and to betroth his daughter Sansa to Robert's son Joffrey. Ned accepts the position when he learns that Arryn's widow Lysa believes he was poisoned by Robert's wife Queen Cersei Lannister and her family. Shortly thereafter, Ned's son Bran discovers Cersei having sex with her twin brother Jaime Lannister, who throws Bran from the tower to conceal their affair, leaving him comatose and paralyzing his legs.

Ned leaves his castle, Winterfell, and departs for the capital city, King's Landing, bringing along his daughters Sansa and Arya. Upon arriving in King's Landing to take his post as Hand, Ned finds that Robert is an ineffective king whose only interests are hunting, drinking, and womanizing.

At Winterfell, an assassin attempts to kill Bran while he is unconscious, and Ned's wife Catelyn travels to King's Landing to bring word to Ned. Catelyn's childhood friend, Petyr "Littlefinger" Baelish, implicates Tyrion Lannister, the dwarf brother of Cersei and Jaime, in the assassination attempt. On the road back to Winterfell, Catelyn encounters Tyrion by chance, arrests him, and takes him to the Vale, where her sister Lysa Arryn is regent, to stand trial for the attempt on Bran's life. In retaliation for Tyrion's abduction, his father Lord Tywin Lannister sends soldiers to raid the Riverlands, Catelyn's home region. Tyrion regains his freedom by recruiting a mercenary named Bronn to defend him in trial by combat.

Ned investigates Jon Arryn's death and eventually discovers that Robert's legal heirs, including Joffrey, are in fact Cersei's children by Jaime (making Robert's uncharismatic younger brother Stannis the rightful heir to the Iron Throne), and that Jon Arryn was killed to conceal his discovery of their incest. Ned offers Cersei a chance to flee before he informs Robert, but she uses this chance to arrange Robert's death in a hunting “accident” and install Joffrey on the throne. Ned prepares to send his daughters away from King's Landing and enlists Littlefinger's help to challenge Joffrey's claim; but Littlefinger betrays him, resulting in Ned's arrest. Arya escapes the castle, but Sansa is taken hostage by the Lannisters.

Ned's eldest son Robb marches his army south in response to his father's arrest, and in order to relieve the threat on the riverlands. To secure a strategically necessary bridge crossing, Catelyn negotiates a marital alliance between Robb and the notoriously unreliable House Frey. Robb defeats a Lannister army in the Riverlands, capturing Jaime. Tywin sends Tyrion back to King's Landing to act as Hand of the King to Joffrey. When Ned is executed, Robb's followers declare the North's independence from the Seven Kingdoms, proclaiming Robb "King in the North".

On the Wall
The prologue of the novel introduces the Wall: an ancient barrier of stone, ice, and magic, hundreds of feet high and hundreds of miles long, shielding the Seven Kingdoms from the northern wilderness. The Wall is defended by the Night's Watch: an order of warriors sworn to serve there for life, defending the realm from the fabled Others, an ancient and hostile inhuman race, as well as from the human "wildlings" who live north of the Wall.

Jon Snow, Ned's bastard son, is inspired by his uncle, Benjen Stark, to join the Night's Watch, but becomes disillusioned when he discovers that its primary function is as a penal colony. Jon unites his fellow recruits against their harsh instructor and protects the cowardly but good-natured and intelligent Samwell Tarly. Jon is appointed steward to the leader of the Watch, Lord Commander Jeor Mormont, making him a potential successor to Mormont. Benjen fails to return from an expedition north of the Wall. Six months later, the dead bodies of two men from his party are recovered; these re-animate as undead wights before being dispatched by Jon.

When word of his father's execution reaches Jon, he attempts to join Robb against the Lannisters, but is persuaded to remain loyal to the Watch. Mormont then declares his intention to march north to find Benjen, dead or alive, and to investigate rumors of a "King-beyond-the-Wall" uniting the wildlings.

Across the Narrow Sea
Across the sea to the east of Westeros live the exiled prince Viserys and princess Daenerys, children of the late "Mad King" Aerys Targaryen, who ruled Westeros before being overthrown by Robert Baratheon. Viserys betroths Daenerys to Khal Drogo, a warlord of the nomadic Dothraki people, in exchange for the use of Drogo's army to reclaim the throne of Westeros. Illyrio Mopatis, a wealthy merchant who has been supporting the penniless Targaryens, gives Daenerys three petrified dragon eggs as a wedding gift. Jorah Mormont, a knight exiled from Westeros, joins Viserys as an adviser. Initially terrified of her new husband and his people, Daenerys eventually embraces her role as Drogo's "khaleesi ". Drogo, however, shows little interest in conquering Westeros, and an impatient Viserys tries to browbeat his sister into coercing Drogo. When Viserys publicly threatens Daenerys and her unborn child, Drogo executes him by pouring molten gold on his head.

An assassin seeking King Robert's favor attempts to poison Daenerys, finally convincing Drogo to conquer Westeros. While sacking villages to fund the invasion of Westeros, Drogo is badly wounded, and Daenerys commands the captive folk healer Mirri Maz Duur to save him. The healer, angered by the Dothraki raids against her people, sacrifices Daenerys's unborn child to power the spell to save Drogo's life, which restores Drogo's physical health but leaves him in a persistent vegetative state.

With Drogo completely incapacitated and unable to lead, much of the Dothraki army disperses. Daenerys smothers Drogo with a pillow and has Mirri tied to Drogo's funeral pyre. She places her three dragon eggs on the pyre and enters it herself. When the fire burns out, she emerges unharmed, with three newly hatched dragons. Awe-struck, Jorah and the remaining Dothraki swear allegiance to her.

"""

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
messages = [{'role': 'user', 'content': input_text}]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([prompt], return_tensors="pt")

input_ids = inputs['input_ids'].to('cuda')
attention_mask = inputs['attention_mask'].to('cuda')

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
    # eos_token_id=128009, 
    # pad_token_id=128009
)

outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,  
    max_new_tokens=500, 
    use_cache=True, 
    return_dict_in_generate=True, 
    output_scores=True, 
    generation_config=gen_config
)

print ("=====================")
input_length = inputs["input_ids"].shape[1]
generated_tokens = outputs.sequences[:, input_length:]
print (generated_tokens.tolist())
print (tokenizer.decode(generated_tokens.tolist()[0]))
