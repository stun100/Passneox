from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
print(model.config.vocab_size)
# Beginning of sequence
bos_token = tokenizer.bos_token or ""
input_text = "My name"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate text
n_steps = 10  # Number of tokens to generate
for _ in range(n_steps):
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)

# Decode the full generated sequence
output_text = tokenizer.decode(input_ids[0])
print("Generated text:", output_text)

