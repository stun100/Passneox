# Passneox

## 📥 Installation

### 1. Install `uv`

If you don't already have `uv` installed:

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

### 2. Initialize uv project
```bash
uv init
```

### 3. Create and Activate Virtual Environment

```bash
uv venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

### 4. Install Dependencies

```bash
uv sync 
```

To enable CUDA
```bash
uv sync --extra cu128
```

---

## ▶️ Usage
```bash
python run_model.py --model_path gpt_neox_multiseq/final_model/ --tokenizer_path tokenizer/ --output_path output/
```