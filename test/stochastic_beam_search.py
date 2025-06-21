import torch
import numpy as np
from torch.distributions import Gumbel
from torch.nn import functional as F
 
def stochastic_beam_search(model, tokenizer, input_ids, k=5, max_steps=20):
    device = input_ids.device
    vocab_size = model.config.vocab_size
 
    # Initialize BEAM: (sequence, log_prob, gumbel_score)
    beam = [(input_ids.clone(), 0.0, 0.0)]
 
    for step in range(max_steps):
        expansions = []
 
        for seq, phi_s, g_phi_s in beam:
            # Get model predictions for this sequence
            with torch.no_grad():
                outputs = model(seq)
                logits = outputs.logits[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
 
            # Find Z (maximum Gumbel score among children)
            Z = float('-inf')
            gumbel_scores = {}
 
            # Sample Gumbel for all possible next tokens
            gumbel_dist = Gumbel(0, 1)
            for token_id in range(vocab_size):
                phi_s_prime = phi_s + log_probs[0, token_id].item()
                g_phi_s_prime = phi_s_prime + gumbel_dist.sample().item()
                gumbel_scores[token_id] = g_phi_s_prime
                Z = max(Z, g_phi_s_prime)
 
            # Create expansions with truncated Gumbel scores
            for token_id in range(vocab_size):
                phi_s_prime = phi_s + log_probs[0, token_id].item()
                g_phi_s_prime = gumbel_scores[token_id]
 
                # Compute truncated Gumbel score
                g_tilde = -torch.log(
                    torch.exp(-torch.tensor(g_phi_s)) - 
                    torch.exp(-torch.tensor(Z)) + 
                    torch.exp(-torch.tensor(g_phi_s_prime))
                ).item()
 
                # Create new sequence
                new_seq = torch.cat([seq, torch.tensor([[token_id]], device=device)], dim=1)
                expansions.append((new_seq, phi_s_prime, g_tilde))
 
        # Keep top k expansions
        expansions.sort(key=lambda x: x[2], reverse=True)  # Sort by g_tilde
        beam = expansions[:k]
 
        # Check for termination (if EOS token generated)
        if any(seq[0, -1].item() == tokenizer.eos_token_id for seq, _, _ in beam):
            break
 
    return beam