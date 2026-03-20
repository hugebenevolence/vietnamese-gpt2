# Vietnamese GPT-2 Pre-training

Pre-train GPT-2 from scratch on Vietnamese text data (BKAI News + Wikipedia), with optional supervised fine-tuning (SFT) for poetry generation.

## Requirements

- Python >= 3.11
- CUDA GPU (recommended for training)

## Setup

```bash
uv sync
# or: pip install -r requirements.txt
```

`requirements.txt` and `pyproject.toml` list the same dependencies for pip and uv.

## Project Structure

```
config.py                 # Hyperparameters and paths
train.py                  # Pre-training (DDP-capable)
inference.py              # Text generation
sft_poem.py               # SFT for poem generation
generate_poem.py          # Generate poems from SFT checkpoint
tokenizer.py              # Train BPE tokenizer
utils.py                  # Shared helpers (text norm, GPT-2 load helper, logging setup)
data_prep/
  news/download_datasets.py    # Download BKAI news corpus → data/train/
  wiki/crawl_vi_wiki.py        # Crawl Vietnamese Wikipedia → data/raws/
  wiki/process_vi_wiki.py      # Clean wikitext → data/raws/ + data/train/
  poem/crawl_poem.py           # Poem metadata (thivien.net)
  poem/scrape_poem_content.py  # Poem full text
  poem/prepare_poem_data.py    # JSONL stanzas for SFT
data/                     # Outputs: raws/, train/, sft/ (see .gitignore)
scripts/
  train.sh
  train_sft_poem.sh
artifacts/
  tokenizer/
  checkpoints/
```

## Pre-training Pipeline

### 1. Prepare Data

```bash
python data_prep/news/download_datasets.py
python data_prep/wiki/crawl_vi_wiki.py
python data_prep/wiki/process_vi_wiki.py
```

### 2. Train Tokenizer

```bash
python tokenizer.py
```

### 3. Train Model

```bash
# Single GPU
python train.py

# Multi-GPU
bash scripts/train.sh
```

### 4. Inference

```bash
python inference.py
```

## Poem SFT Pipeline

### 1. Prepare Poem Data

```bash
python data_prep/poem/crawl_poem.py
python data_prep/poem/scrape_poem_content.py
python data_prep/poem/prepare_poem_data.py
```

### 2. Train SFT

```bash
bash scripts/train_sft_poem.sh
```

### 3. Generate Poems

```bash
python generate_poem.py
```

## Configuration

All settings live in `config.py`. Pre-training resumes from the latest checkpoint under `CHECKPOINT_DIR`.
