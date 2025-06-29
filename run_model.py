import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations for reproducibility
import yaml
import argparse
import logging
import torch
import numpy as np
import pandas as pd
import time
from tokenizers import SentencePieceBPETokenizer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, GPTNeoXConfig
from typing import Set, Tuple
from sbs_batched import StochasticBeamSearch
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def linear_schedule(start: float, end: float, num_points: int) -> np.array:
    """Generate a linear schedule from start to end."""
    return np.linspace(start, end, num_points)

def inverse_exponential_schedule(start: float, end: float, num_points: int, multiplier: float = 1.0) -> np.array:
    """Generate an inverse exponential schedule from start to end."""
    return start + (end - start) * (1 - np.exp(-np.linspace(0, 5, num_points))) * multiplier

def generate_outputs(
    model_path: str,
    tokenizer_path: str, 
    num_outputs: int = 10, 
    batch_size: int = 500,
    max_password_length: int = 12,
    top_k: int = 50, 
    top_p: float = 0.95,
    initial_temperature: float = 0.7,
    penalty_alpha: float = 0.0,
    max_temperature: float = 2.0,
    strings_per_sequence: int = 1,
    use_contrastive_search: bool = False,
    use_sbs: bool = False,
    sbs_k: int = 3
) -> Tuple[Set[str], int, pd.DataFrame]:
    """Generate outputs using the model and tokenizer."""
    start_time = time.time()
    
    # Load tokenizer
    logging.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = SentencePieceBPETokenizer.from_file(
        vocab_filename=f"{tokenizer_path}/vocab.json",
        merges_filename=f"{tokenizer_path}/merges.txt"
    )
    
    # Define special tokens
    sep_token = "<sep>"
    pad_token = "<pad>"
    eos_token = "</s>"
    bos_token = "<s>"
    unk_token = "<unk>"
    
    # Get token IDs
    pad_token_id = tokenizer.token_to_id(pad_token) or 0
    eos_token_id = tokenizer.token_to_id(eos_token) or 1
    bos_token_id = tokenizer.token_to_id(bos_token) or 0
    sep_token_id = tokenizer.token_to_id(sep_token)
    
    # Load model
    logging.info(f"Loading model from {model_path}")
    config = GPTNeoXConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    model.to(device).eval()
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {total_params:,}")
    
    # Prepare input
    input_encoding = tokenizer.encode(bos_token)
    input_ids = torch.tensor([input_encoding.ids], dtype=torch.long)
    print(input_ids.shape)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(device)
    
    # Calculate max token length
    adjusted_max_token_length = max_password_length * strings_per_sequence + strings_per_sequence
    
    # Estimate number of batches needed
    estimated_strings_per_batch = batch_size * strings_per_sequence
    num_batches = (num_outputs + estimated_strings_per_batch - 1) // estimated_strings_per_batch
    
    logging.info(f"Generating {num_outputs} total strings in ~{num_batches} batches")
    
    # Create inverse exponential temperature schedule
    max_possible_batches = num_batches * 2  # Double to allow for extra batches if needed
    temperature_schedule = inverse_exponential_schedule(initial_temperature, max_temperature, max_possible_batches)
    
    all_generated_strings = []
    batch_stats = []
    

    sbs = StochasticBeamSearch(k=sbs_k, steps=adjusted_max_token_length -1, device=device, eos_token_id=eos_token_id)

    input_ids = input_ids.repeat(batch_size, 1)
    with torch.no_grad():
        batch_num = 0
        
        while len(all_generated_strings) < num_outputs:
            batch_num += 1
            batch_start = time.time()
            
            # Get current temperature
            temp_idx = min(batch_num - 1, len(temperature_schedule) - 1)
            current_temp = temperature_schedule[temp_idx]
            
            # Calculate how many sequences we need for this batch
            strings_remaining = num_outputs - len(all_generated_strings)
            sequences_needed = min(batch_size, (strings_remaining + strings_per_sequence - 1) // strings_per_sequence)
            
            try:
                if use_contrastive_search:

                    # Generate sequences
                    generated_outputs = model.generate(
                        input_ids,
                        max_length=adjusted_max_token_length,
                        num_return_sequences=sequences_needed,
                        #do_sample=True,
                        top_k=top_k,
                        #top_p=top_p,
                        #temperature=current_temp,
                        attention_mask=attention_mask,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                        penalty_alpha=penalty_alpha
                    )
                elif use_sbs:
                    
                    generated_outputs = sbs.search(model, input_ids, attention_mask)
                
                else:
                    # Generate sequences
                    generated_outputs = model.generate(
                        input_ids,
                        max_length=adjusted_max_token_length,
                        num_return_sequences=sequences_needed,
                        do_sample=True,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=current_temp,
                        attention_mask=attention_mask,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                        penalty_alpha=penalty_alpha
                    )


                
                # Process outputs
                batch_strings = []
                for output in generated_outputs:
                    try:
                        decoded = tokenizer.decode(output.tolist())
                        # Clean up special tokens
                        cleaned = (decoded.replace(bos_token, '')
                                 .replace(eos_token, '')
                                 .replace(pad_token, '')
                                 .replace(unk_token, ''))
                        
                        # Split by separator and get valid strings
                        if sep_token_id and sep_token in cleaned:
                            strings = [s.strip() for s in cleaned.split(sep_token) if s.strip()]
                        else:
                            strings = [cleaned.strip()] if cleaned.strip() else []
                        
                        batch_strings.extend(strings)
                    except Exception as e:
                        logging.warning(f"Error decoding sequence: {e}")
                        continue
                
                # Add to total collection
                all_generated_strings.extend(batch_strings)
                
                # Trim if we have too many
                if len(all_generated_strings) > num_outputs:
                    all_generated_strings = all_generated_strings[:num_outputs]
                
                # Track batch statistics
                batch_unique = len(set(batch_strings))
                batch_duplicates = len(batch_strings) - batch_unique
                batch_time = time.time() - batch_start
                
                batch_stats.append({
                    'batch': batch_num,
                    'temperature': current_temp,
                    'strings_generated': len(batch_strings),
                    'batch_duplicates': batch_duplicates,
                    'batch_time': batch_time
                })
                
                # Progress logging
                unique_so_far = len(set(all_generated_strings))
                total_so_far = len(all_generated_strings)
                logging.info(f"Batch {batch_num}: {len(batch_strings)} strings (temp={current_temp:.2f}) | "
                           f"Total: {total_so_far}/{num_outputs} | Unique: {unique_so_far}")
                
            except Exception as e:
                logging.error(f"Error in batch {batch_num}: {e}")
                continue
    
    # Ensure we have exactly num_outputs
    if len(all_generated_strings) < num_outputs:
        # Fill remaining with duplicates of last string if needed
        missing = num_outputs - len(all_generated_strings)
        if all_generated_strings:
            all_generated_strings.extend([all_generated_strings[-1]] * missing)
        else:
            all_generated_strings = ["placeholder"] * num_outputs
        logging.warning(f"Filled {missing} missing strings")
    
    # Final results
    unique_strings = set(all_generated_strings)
    duplicate_count = len(all_generated_strings) - len(unique_strings)
    
    # Create stats dataframe
    if batch_stats:
        stats_df = pd.DataFrame(batch_stats)
    else:
        stats_df = pd.DataFrame()
    
    total_time = time.time() - start_time
    logging.info(f"Generation complete: {len(unique_strings)} unique, {duplicate_count} duplicates "
                f"({duplicate_count/len(all_generated_strings)*100:.1f}%) in {total_time:.2f}s")
    
    return unique_strings, duplicate_count, stats_df

def write_outputs(sequences: Set[str], output_path: str) -> None:
    """Write the generated sequences to a file."""
    output_file_path = os.path.join(output_path, 'out.txt')
    with open(output_file_path, 'w') as f:
        for sequence in sequences:
            try:
                f.write(sequence + '\n')
            except Exception:
                continue  # Skip sequences with encoding issues
    logging.info(f"Outputs written to {output_file_path}")

if __name__ == '__main__':
    try:
        # Load config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract parameters
        gen_config = config['generator']
        num_outputs = gen_config['num_outputs']
        batch_size = gen_config['batch_size']
        max_password_length = config['global']['max_password_length']
        top_k = gen_config['top_k']
        top_p = gen_config['top_p']
        initial_temperature = gen_config['temperature']
        penalty_alpha = gen_config['penalty_alpha']
        max_temperature = gen_config['max_temperature']
        strings_per_sequence = gen_config['strings_per_sequence']
        use_contrastive_search = gen_config.get('use_contrastive_search', False)
        use_sbs = gen_config.get('use_sbs', False)
        sbs_k = gen_config['sbs_k']

        # Parse arguments
        parser = argparse.ArgumentParser(description="Generate passwords using pretrained model")
        parser.add_argument('--model_path', type=str, default='model/gpt_neox_multiseq', help='Path to model')
        parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to tokenizer files')
        parser.add_argument('--output_path', type=str, required=True, help='Output directory')
        parser.add_argument('--contrastive', action='store_true', help='Use contrastive search instead of sampling')
        parser.add_argument('--sbs', action='store_true', help='Use Stochastic Beam Search')
        parser.add_argument('--sampling', action='store_true', help='Use sampling instead of contrastive search')
        args = parser.parse_args()
        
        # Override config with command line arguments
        if args.contrastive:
            use_contrastive_search = True
            use_sbs = False
        elif args.sbs:
            use_contrastive_search = False
            use_sbs = True
        else:
            use_contrastive_search = False
            use_sbs = False
        
        logging.info(f"Generating {num_outputs} passwords")
        logging.info(f"Model: {args.model_path} | Tokenizer: {args.tokenizer_path}")
        if use_contrastive_search:
            logging.info(f"Using Contrastive Search as deconding method")
        # Generate passwords
        unique_sequences, duplicate_count, stats = generate_outputs(
            args.model_path, 
            args.tokenizer_path,
            num_outputs=num_outputs,
            batch_size=batch_size,
            max_password_length=max_password_length, 
            top_k=top_k, 
            top_p=top_p, 
            initial_temperature=initial_temperature,
            penalty_alpha=penalty_alpha,
            max_temperature=max_temperature,
            strings_per_sequence=strings_per_sequence,
            use_contrastive_search=use_contrastive_search,
            use_sbs=use_sbs,
            sbs_k=sbs_k
        )
        
        # Save results
        os.makedirs(args.output_path, exist_ok=True)
        write_outputs(unique_sequences, args.output_path)
        
        # Final summary
        logging.info(f"Complete: {len(unique_sequences)} unique passwords, {duplicate_count} duplicates")
        logging.info(f"Duplicate rate: {duplicate_count/num_outputs*100:.1f}%")
        
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        logging.error(traceback.format_exc())
