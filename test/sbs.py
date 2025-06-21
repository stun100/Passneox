import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, MinLengthLogitsProcessor
from torch.distributions import Gumbel
import timeit

class StochasticBeamSearch:
    """
    Implements Stochastic Beam Search (SBS) from the paper:
    "Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for
    Sampling Sequences Without Replacement" (Kool et al., 2019).
    
    This class is designed to be compatible with Hugging Face's `transformers` models.
    """
    def __init__(self, k, steps, device, eos_token_id):
        self.k = k
        self.steps = steps
        self.device = device
        self.eos_token_id = eos_token_id

    def search(self, model, input_ids):
        device = input_ids.device
        vocab_size = model.config.vocab_size
        # Initialize beams: (sequence, log_prob, gumbel_score)
        beams = [(input_ids.clone(), 0, 0)]
        
        # Generate tokens step by step
        for t in range(self.steps):
            expansions = []
            
            # Expand each beam by considering all possible next tokens
            for beam in beams:
                seq, phi_s, g_phi_s = beam
                Z = float('-inf')  # Track maximum Gumbel score
                gumbel_scores = {}
                
                # Get model predictions for next token
                with torch.no_grad():
                    outputs = model(seq)
                    logits = outputs.logits[:, -1, :]  # Last token logits
                    log_probs = F.log_softmax(logits, dim=-1)
                
                # First pass: compute Gumbel scores for all tokens
                for token_id in range(vocab_size):
                    phi_s_prime = phi_s + log_probs[0, token_id].item()  # Add log probabilities to phi_s
                    gumbel_distr = Gumbel(phi_s_prime, 1)
                    g_phi_S_prime = phi_s_prime + gumbel_distr.sample().item()
                    gumbel_scores[token_id] = g_phi_S_prime
                    Z = max(Z, g_phi_S_prime)
                
                # Second pass: compute adjusted Gumbel scores (g_tilde)
                for token_id in range(vocab_size):
                    phi_s_prime = phi_s + log_probs[0, token_id].item()
                    g_phi_s_prime = gumbel_scores[token_id]

                    # Compute g_tilde using the SBS formula
                    g_tilde = -torch.log(
                        torch.exp(-torch.tensor(g_phi_s)) - 
                        torch.exp(-torch.tensor(Z)) + 
                        torch.exp(-torch.tensor(g_phi_s_prime))
                    ).item()

                    # Create expanded sequence
                    y_s_prime = torch.cat([seq, torch.tensor([[token_id]], device=device)], dim=1)
                    expansions.append((y_s_prime, phi_s_prime, g_tilde))

            # Select top-k beams based on adjusted Gumbel scores
            expansions.sort(key=lambda x: x[2], reverse=True)
            beams = expansions[:self.k]
    
        return beams
    
if __name__ == "__main__":
    # Example usage
    model_name = "gpt2"
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    input_text = "My name is"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    sbs = StochasticBeamSearch(k=1, steps=1, device=device, eos_token_id=tokenizer.eos_token_id)
    print("Run search")
    start_time = timeit.default_timer()
    beams = sbs.search(model, input_ids)
    elapsed = timeit.default_timer() - start_time
    print(f"Search time: {elapsed:.4f} seconds")

    print("Generated sequences:")
    for seq, phi_s, g_tilde in beams:
        output_text = tokenizer.decode(seq[0], skip_special_tokens=True)
        print(f"Sequence: {output_text}")