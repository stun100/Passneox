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
        beams = [(input_ids.clone(), torch.zeros((batch_size,1), device=self.device), torch.zeros((batch_size,1), device=self.device))]
        # Generate tokens step by step
        for t in range(self.steps):
            expansions = []
            
            # Expand each beam by considering all possible next tokens
            for i, beam in enumerate(beams):
                # seq shape: (b, seq_len)
                seq, phi_s, g_phi_s = beam
                Z = torch.tensor([float('-inf')], device=self.device)  # Track maximum Gumbel score
                Z = Z.repeat(input_ids.shape[0], 1) # (b, 1)

                # Get model predictions for next token
                with torch.no_grad():
                    outputs = model(seq, attention_mask=attention_mask)
                    logits = outputs.logits[:, -1, :]  # Last token logits
                    log_probs = F.log_softmax(logits, dim=-1) # (b, |V|)
                
                gumbel_distr = Gumbel(torch.zeros(log_probs.shape, device=self.device), 1, ) # Gumbel distribution for sampling

                phi_s_prime = phi_s + log_probs # (b, |V|)
                g_phi_s_prime = phi_s_prime + gumbel_distr.sample() # (b, |V|)

                Z = torch.max(
                    torch.cat([
                        g_phi_s_prime,
                        Z     
                    ], dim=1) # (b, V + 1)
                , dim=1, keepdim=True).values # (b, 1)


                Zb = Z.expand_as(g_phi_s_prime)  # (batch, V)

                delta = g_phi_s_prime - Zb       # (batch, V)

                log1mexp = torch.where(
                    delta > -0.693,                   
                    torch.log(-torch.expm1(delta)),   
                    torch.log1p(-torch.exp(delta))    
                )  # (batch, V)

                v = g_phi_s - g_phi_s_prime + log1mexp  # (batch, V)

                g_tilde = (
                    g_phi_s
                    - torch.maximum(v, torch.zeros_like(v))
                    - torch.log1p(torch.exp(-v.abs()))
                )  # (batch, V)

                # g_tilde = -torch.log(torch.clamp(
                #     torch.exp(-g_phi_s) - 
                #     torch.exp(-Z) + 
                #     torch.exp(-g_phi_s_prime), min=1e-9)
                # ) # (b, |V|)
                # if float('inf') in g_tilde:
                #     print(t)

                # seq = (b, seq_len)
                # y_s_prime = (b, seq_len + 1, |V|)
                seq_repeated = seq.unsqueeze(0).permute(1,0,2).repeat(1, vocab_size, 1) # (b, V, seq_len)
                vocab_tokens = torch.arange(0, vocab_size, device=self.device).repeat(batch_size, 1) # (b, V)
                y_s_prime = torch.cat([seq_repeated, vocab_tokens.unsqueeze(-1)], dim=2) # (b, |V|, seq_len + 1))

                # Expansions[0]: (b, V, seq_len + 1 + 1 + 1)
                expansions.append(torch.cat([y_s_prime, phi_s_prime.unsqueeze(-1), g_tilde.unsqueeze(-1)], dim=2)) 
      
            stacked_expansion = torch.stack(expansions).permute(1,0,2,3)

            batch_size, k_dim, v_dim, data_dim = stacked_expansion.shape

            g_tildes = stacked_expansion[:, :, :, -1]  # (b, k, v)

            g_tildes_flat = g_tildes.reshape(batch_size, -1)  # (b, k*v)
            topk_values, topk_indices = torch.topk(g_tildes_flat, k=self.k, dim=1)  # (b, k)

            topk_k_indices = topk_indices // v_dim  # (b, k)
            topk_v_indices = topk_indices % v_dim   # (b, k)

            batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, self.k)  # (b, k)

            # (k, b, seq_len + 1 + 1 + 1)
            topk_vectors = stacked_expansion[batch_indices, topk_k_indices, topk_v_indices].permute(1,0,2)

            seq_length = data_dim - 2
            beams = [(topk_vectors[i, :, :seq_length].int(), 
                      topk_vectors[i, :, seq_length:seq_length+1], 
                      topk_vectors[i, :, -1].unsqueeze(-1)) for i in range(topk_vectors.shape[0])]
            
            output = beams[0][0]
        return output
    
if __name__ == "__main__":
    # Example usage
    model_name = "gpt2"
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    input_text = ["My name is", "I love eating"]
    input_ids = tokenizer.encode(input_text, return_tensors="pt").view(2,3)
    
    sbs = StochasticBeamSearch(k=3, steps=10, device=device, eos_token_id=tokenizer.eos_token_id)
    print("Run search")
    start_time = timeit.default_timer()
    beams = sbs.search(model, input_ids)
    elapsed = timeit.default_timer() - start_time
    print(f"Search time: {elapsed:.4f} seconds")

    print("Generated sequences:", beams)
    for seq, phi_s, g_tilde in beams:
        for batch in range(seq.shape[0]):
            output_text = tokenizer.decode(seq[batch], skip_special_tokens=True)
            print(f"Sequence: {output_text}")