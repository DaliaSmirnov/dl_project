from transformers import AutoModelWithLMHead, AutoTokenizer
import torch

model_robin = AutoModelWithLMHead.from_pretrained('output-medium-robin')
model_barney = AutoModelWithLMHead.from_pretrained('output-medium-barney')
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
model = AutoModelWithLMHead.from_pretrained('output-medium')


seed = 'Lets go to the strip club'
seed_input_ids = tokenizer.encode(seed + tokenizer.eos_token, return_tensors='pt')
barney_input_ids = seed_input_ids

barney_chat_history_ids = model_barney.generate(
        barney_input_ids,
        max_length=1000,
        do_sample=True,
        top_p=0.95,
        top_k=0,
        temperature=0.75,
        pad_token_id=tokenizer.eos_token_id
    )
barneys_answer = tokenizer.decode(barney_chat_history_ids[:, barney_input_ids.shape[-1]:][0], skip_special_tokens=True)
print(f"Seed: {seed}")
print(f"Barney: {barneys_answer}")


for step in range(8):
      new_barney_input_ids = tokenizer.encode(barneys_answer + tokenizer.eos_token, return_tensors='pt')
      robin_input_ids = torch.cat([robin_chat_history_ids, new_barney_input_ids], dim=-1) if step > 0 else new_barney_input_ids
      robin_chat_history_ids = model_robin.generate(
        robin_input_ids,
        max_length=1000,
        do_sample=True,
        top_p=0.95,
        top_k=0,
        temperature=0.75,
        pad_token_id=tokenizer.eos_token_id
    )
      robins_answer = tokenizer.decode(robin_chat_history_ids[:, robin_input_ids.shape[-1]:][0], skip_special_tokens=True)
      print(f"Robin: {robins_answer}")



      new_robin_input_ids = tokenizer.encode(robins_answer + tokenizer.eos_token, return_tensors='pt')
      barney_input_ids = torch.cat([barney_chat_history_ids, new_robin_input_ids], dim=-1)
      barney_chat_history_ids = model_barney.generate(
        barney_chat_history_ids,
        max_length=1000,
        do_sample=True,
        top_p=0.95,
        top_k=0,
        temperature=0.75,
        pad_token_id=tokenizer.eos_token_id
    )
      barneys_answer = tokenizer.decode(barney_chat_history_ids[:, barney_input_ids.shape[-1]:][0], skip_special_tokens=True)
      print(f"Barney: {barneys_answer}")