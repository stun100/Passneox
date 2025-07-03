#!/bin/bash

# Paths
MODEL_PATH="gpt_neox_multiseq/final_model/"
TOKENIZER_PATH="tokenizer/"
OUTPUT_PATH="outputs/"

# Config files
CONFIG_FILES=("config_1.yaml")

# Loop through each config file and run the model
for CONFIG in "${CONFIG_FILES[@]}"; do
  echo "Running with config: $CONFIG"
  python run_model.py \
    --config_file "$CONFIG" \
    --model_path "$MODEL_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --output_path "$OUTPUT_PATH" \
    --sbs 
done
