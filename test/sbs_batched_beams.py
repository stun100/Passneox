import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
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

    def search(self, model, input_ids, attention_mask):
        vocab_size = model.config.vocab_size
        batch_size = input_ids.shape[0]
        # Initialize beams: (sequence, log_prob, gumbel_score)
        seqs = input_ids.unsqueeze(1).expand(-1, self.k, -1).clone() # (b, k, seq_len)
        phi_s = torch.zeros(batch_size, self.k, device=self.device) # (b, k)
        g_phi_s = torch.zeros(batch_size, self.k, device=self.device) # (b, k)

        # beams = [(input_ids.clone(), torch.zeros((batch_size,1), device=self.device), torch.zeros((batch_size,1), device=self.device))]
        # Generate tokens step by step
        for t in range(self.steps):
            expansions = []

            flat_seqs = seqs.reshape(batch_size * self.k, -1) # (b*k, seq_len)
            
            # Expand each beam by considering all possible next tokens
            # # seq shape: (b, seq_len)
            # Z = torch.tensor([float('-inf')], device=self.device)  # Track maximum Gumbel score
            # Z = Z.repeat(input_ids.shape[0], 1) # (b, 1)

            # Get model predictions for next token
            with torch.no_grad():
                outputs = model(flat_seqs)#, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :]  # Last token logits
                log_probs = F.log_softmax(logits, dim=-1) # (b*k, |V|)
            
            log_probs = log_probs.view(batch_size, self.k, vocab_size) # (b, k, V)
            
            gumbel_distr = Gumbel(torch.zeros_like(log_probs, device=self.device), 1) # Gumbel distribution for sampling

            phi_s_prime = phi_s.unsqueeze(-1) + log_probs # (b, k, |V|)
            g_phi_s_prime = phi_s_prime + gumbel_distr.sample() # (b, k, |V|)

            Z = torch.max(g_phi_s_prime, dim=2, keepdim=True).values # (b, k, 1)

            Zb = Z.expand_as(g_phi_s_prime)  # (batch, k, V)

            delta = g_phi_s_prime - Zb       # (batch, k, V)

            log1mexp = torch.where(
                delta > -0.693,                   
                torch.log(-torch.expm1(delta)),   
                torch.log1p(-torch.exp(delta))    
            )  # (batch, k, V)

            v = g_phi_s.unsqueeze(-1) - g_phi_s_prime + log1mexp  # (batch, k, V)

            g_tilde = (
                g_phi_s.unsqueeze(-1)
                - torch.maximum(v, torch.zeros_like(v))
                - torch.log1p(torch.exp(-v.abs()))
            )  # (batch, k, V)

            # g_tilde = -torch.log(torch.clamp(
            #     torch.exp(-g_phi_s) - 
            #     torch.exp(-Z) + 
            #     torch.exp(-g_phi_s_prime), min=1e-9)
            # ) # (b, |V|)
            # if float('inf') in g_tilde:
            #     print(t)

            # seqs = (b, k, seq_len)
            seq_repeated = seqs.unsqueeze(2).expand(-1, -1, vocab_size, -1) # (b, k, V, seq_len)
            vocab_tokens = torch.arange(0, vocab_size, device=self.device).view(1, 1, vocab_size).expand(batch_size, self.k, -1) # (b,k,V)
            y_s_prime = torch.cat([seq_repeated, vocab_tokens.unsqueeze(-1)], dim=3) # (b, k, |V|, seq_len + 1))

            g_tildes_flat = g_tilde.view(batch_size, -1)  # (b, k*v)
            topk_values, topk_indices = torch.topk(g_tildes_flat, k=self.k, dim=1)  # (b, k)

            beam_indices = topk_indices // vocab_size  # (b, k)
            token_indices = topk_indices % vocab_size   # (b, k)

            batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, self.k)  # (b, k)

            # (k, b, seq_len + 1 + 1 + 1)
            #topk_vectors = stacked_expansion[batch_indices, topk_k_indices, topk_v_indices].permute(1,0,2)
            seqs = y_s_prime[batch_indices, beam_indices, token_indices]
            phi_s = phi_s_prime[batch_indices, beam_indices, token_indices]
            g_phi_s = g_tilde[batch_indices, beam_indices, token_indices]
            
        return seqs[:, 0, :]
    
if __name__ == "__main__":
    # Example usage
    model_name = "gpt2"
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    input_text = ["My name is", "I love eating"]
    input_ids = tokenizer.encode(input_text, return_tensors="pt").view(2,3).to(device)
    attention_mask = torch.ones((input_ids.shape[0]*3, input_ids.shape[1]), dtype=torch.long).to(device)

    sbs = StochasticBeamSearch(k=3, steps=10, device=device, eos_token_id=tokenizer.eos_token_id)
    print("Run search")
    start_time = timeit.default_timer()
    outputs = sbs.search(model, input_ids, attention_mask=attention_mask)
    elapsed = timeit.default_timer() - start_time
    print(f"Search time: {elapsed:.4f} seconds")

    print("Generated sequences:")
    for output in outputs:
        output_text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Sequence: {output_text}")