# Passneox

## 📥 Installation

###  0. Download model weights on the Releases section

Put `model.safetensors` under the `gpt_neox_multiseq/final_model` directory 

### 1. Initialize uv project
```bash
uv init
```

### 2. Create and Activate Virtual Environment

```bash
uv venv
source .venv/bin/activate  # On Windows use `.venv/Scripts/activate`
```

### 3. Install Dependencies

```bash
uv sync 
```

To install torch (cpu or CUDA)
```bash
uv sync --extra cpu # cu128
```

---

## ▶️ Usage
```bash
python run_model.py \
  --model_path gpt_neox_multiseq/final_model/ \
  --tokenizer_path tokenizer/ \
  --output_path outputs/
```