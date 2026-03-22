# Vietnamese GPT-2 Pre-training

Pre-train GPT-2 from scratch on Vietnamese data, with optional supervised fine-tuning (SFT) for poetry generation.

## Requirements

- Python >= 3.11
- CUDA GPU

## Setup

```bash
uv sync && uv pip install -e .
# or: pip install -r requirements.txt && pip install -e .
```

Run scripts from the **repository root** (paths in `src/config.py` are relative to cwd).

## Project Structure

```
src/
  config.py           # Hyperparameters and paths
  utils.py            # Text norm, logging, GPT-2 helpers
  train.py            # Pre-training (DDP-capable)
  inference.py        # General text generation
  train_sft.py        # Poem SFT
  generate.py         # Poem generation from SFT checkpoint
  train_tokenizer.py  # Train BPE tokenizer
data_prep/
  news/download_datasets.py
  wiki/crawl_vi_wiki.py
  wiki/process_vi_wiki.py
  deduplicate.py
  poem/...
data/                 # Outputs: raws/, train/, sft/ (see .gitignore)
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
uv run python data_prep/news/download_datasets.py
uv run python data_prep/wiki/crawl_vi_wiki.py
uv run python data_prep/wiki/process_vi_wiki.py
```

### 2. Train Tokenizer

```bash
uv run python src/train_tokenizer.py
```

### 3. Deduplicate

```bash
uv run python data_prep/deduplicate.py                   # with token audit
uv run python data_prep/deduplicate.py --skip-token-audit # faster, no tokenizer needed
```

Outputs deduplicated parquets to `data/train/deduped/` and an audit report
(`data/train/deduped/dedup_report.json`) with per-source stats and token budget analysis.

### 4. Train Model

```bash
bash scripts/train.sh
```

### 5. Inference

```bash
uv run python src/inference.py
```

## Poem SFT Pipeline

### 1. Prepare Poem Data

```bash
uv run python data_prep/poem/crawl_poem.py
uv run python data_prep/poem/scrape_poem_content.py
uv run python data_prep/poem/prepare_poem_data.py
```

### 2. Train SFT

```bash
bash scripts/train_sft_poem.sh
```

### 3. Generate Poems

```bash
uv run python src/generate.py
```

## Configuration

All settings live in `src/config.py`. Pre-training resumes from the latest checkpoint under `CHECKPOINT_DIR`.
