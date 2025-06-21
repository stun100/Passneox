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
        vocab_size = model.config.vocab_size
        # Initialize beams: (sequence, log_prob, gumbel_score)
        beams = [(input_ids.clone(), 0, 0)]
        
        # Generate tokens step by step
        for t in range(self.steps):
            expansions = []
            
            # Expand each beam by considering all possible next tokens
            for i, beam in enumerate(beams):
                # print(t, i)
                seq, phi_s, g_phi_s = beam
                Z = torch.tensor([float('-inf')])  # Track maximum Gumbel score
                
                # Get model predictions for next token
                with torch.no_grad():
                    outputs = model(seq.to(self.device))
                    logits = outputs.logits[:, -1, :].to("cpu")  # Last token logits
                    log_probs = F.log_softmax(logits, dim=-1) # (1, |V|)
                
                gumbel_distr = Gumbel(torch.zeros(log_probs[0,:].shape), 1)# Gumbel distribution for sampling

                # First pass: compute Gumbel scores for all tokens
                phi_s_prime = phi_s + log_probs[0, :]  # (|V|,)
                g_phi_s_prime = phi_s_prime + gumbel_distr.sample() # (|V|,)
                # gumbel_scores = g_phi_s_prime 
                Z = torch.max(
                    torch.cat([
                        g_phi_s_prime,
                        Z     # inherits device & dtype
                    ])
                )

                # Compute g_tilde using the SBS formula
                g_tilde = -torch.log(
                    torch.exp(-torch.tensor(g_phi_s)) - 
                    torch.exp(-Z) + 
                    torch.exp(-g_phi_s_prime)
                ) # (|V|,)

                # seq (1, seq_len)
                # y_s_prime (1, seq_len + 1, |V|)
                seq_repeated = seq.repeat(vocab_size, 1)
                vocab_tokens = torch.arange(0, vocab_size)
                y_s_prime = torch.cat([seq_repeated, vocab_tokens.unsqueeze(1)], dim=1) # (|V|, seq_len + 1))
                
                # Final tensor: (vocab_size, seq_len + 1, 1, 1)
                expansions.append(torch.cat([y_s_prime, phi_s_prime.unsqueeze(1), g_tilde.unsqueeze(1)], dim=1)) 
                # print("exxxxx", expansions[0].shape)
      
            stacked_expansion = torch.stack(expansions)
            # print("stackkkk", torch.stack(expansions)[:,:,].shape)
            # torch.stack(expansions).shape = (k, |V|, seq_len + 1, 1, 1)
            # Select top-k beams based on adjusted Gumbel scores
            topk_values, topk_indices = torch.topk(stacked_expansion[:,:,-1].flatten(), k=self.k)  # Get top k expansions
            topk_coords = torch.stack([topk_indices // stacked_expansion.size(1), topk_indices % stacked_expansion.size(1)], dim=1)
            topk_vectors = stacked_expansion[topk_coords[:, 0], topk_coords[:, 1]]
            print("topklkkkkkkkkkkkkk",topk_vectors.shape)
            # expansions.sort(key=lambda x: x[2], reverse=True)
            # beams = expansions[:self.k]
    
        return beams
    
if __name__ == "__main__":
    # Example usage
    model_name = "gpt2"
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    input_text = "My name is"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    sbs = StochasticBeamSearch(k=3, steps=10, device=device, eos_token_id=tokenizer.eos_token_id)
    print("Run search")
    start_time = timeit.default_timer()
    beams = sbs.search(model, input_ids)
    elapsed = timeit.default_timer() - start_time
    print(f"Search time: {elapsed:.4f} seconds")

    print("Generated sequences:")
    # for seq, phi_s, g_tilde in beams:
    #     output_text = tokenizer.decode(seq[0], skip_special_tokens=True)
    #     print(f"Sequence: {output_text}")