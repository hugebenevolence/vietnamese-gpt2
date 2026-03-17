# Vietnamese GPT-2 Pre-training

Pre-train a GPT-2 language model from scratch on Vietnamese text data (news corpus + Wikipedia).

## Project Structure

```
├── config.py                  # Centralized configuration (paths, hyperparameters)
├── utils.py                   # Shared utilities (text normalization, formatting)
├── train.py                   # Model training script (DDP-compatible)
├── inference.py               # Text generation / interactive inference
├── tokenizer.py               # BPE tokenizer training
├── data/
│   ├── crawl_vi_wiki.py       # Vietnamese Wikipedia crawler (MediaWiki API)
│   ├── process_vi_wiki.py     # Wikitext → clean plaintext pipeline
│   └── download_datasets.py   # Download HuggingFace datasets
├── scripts/
│   └── train.sh               # Multi-GPU training launcher
├── artifacts/
│   ├── tokenizer/             # Trained tokenizer files
│   ├── checkpoints/           # Model checkpoints (gitignored)
│   └── logs/                  # Training logs (gitignored)
└── pyproject.toml             # Project metadata & dependencies
```

## Requirements

- Python >= 3.11
- CUDA-capable GPU(s) recommended for training

## Setup

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

## Data Preparation

### 1. Download datasets

Downloads the BKAI News Corpus from HuggingFace:

```bash
python data/download_datasets.py
```

### 2. Crawl Vietnamese Wikipedia (optional)

```bash
python data/crawl_vi_wiki.py --limit 50000 --delay 1.0
```

### 3. Process Wikipedia data

Cleans wikitext markup and converts to Parquet:

```bash
python data/process_vi_wiki.py
```

## Train Tokenizer

Trains a byte-level BPE tokenizer on the combined dataset:

```bash
python tokenizer.py
```

## Training

### Single GPU

```bash
python train.py
```

### Multi-GPU (DDP)

```bash
bash scripts/train.sh
```

Training automatically resumes from the latest checkpoint if one exists.
All hyperparameters are centralized in `config.py`.

## Inference

Run test examples and optionally enter interactive mode:

```bash
python inference.py
```

## Configuration

All shared paths, hyperparameters, and defaults are in `config.py`.
Edit that file to adjust training parameters, model paths, or dataset configurations.
