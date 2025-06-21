import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "gpt2"  # You can replace with other models like "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# Input prompt
prompt = "Once upon a time in a distant galaxy"

# Encode prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Generate text using different decoding strategies
def decode(strategy, **kwargs):
    print(f"\n--- {strategy.upper()} ---")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=50,
            do_sample=strategy in ["top-k", "top-p"],
            num_beams=kwargs.get("num_beams", 1),
            top_k=kwargs.get("top_k", 50),
            top_p=kwargs.get("top_p", 0.95),
            temperature=kwargs.get("temperature", 1.0),
            early_stopping=True,
            no_repeat_ngram_size=2 if strategy == "beam" else 0,
        )
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

# Test different methods
decode("greedy")                        # Greedy decoding
decode("beam", num_beams=5)            # Beam search
decode("top-k", top_k=50)              # Top-k sampling
decode("top-p", top_p=0.9)             # Nucleus sampling