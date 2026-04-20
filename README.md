# Vietnamese GPT-2: Multi-Stage Pretraining

A clean and reproducible GPT-2 pretraining pipeline for Vietnamese, trained in two stages:

- **Stage 1**: Pretrain GPT-2 from random initialization on mixed Vietnamese corpora.
- **Stage 2**: Continue pretraining on a curated **5-word quatrain poem** corpus for style adaptation.

This repository includes data preparation, tokenizer training, multi-stage pretraining, and text generation scripts.

---

## Repository Structure

```text
vietnamese-gpt2/
├── src/
│   ├── config.py               # Central configuration: paths, datasets, hyperparameters
│   ├── utils.py                # Shared helpers for normalization, callbacks, generation
│   ├── train_tokenizer.py      # Train tokenizer from Vietnamese corpora
│   ├── train_1.py              # Stage 1 pretraining from scratch
│   ├── train_2.py              # Stage 2 continued pretraining on poem corpus
│   ├── generate_base.py        # Generate text with the stage-1 model
│   └── generate_poem.py        # Generate poem-style text with the stage-2 model
├── data_prep/
│   ├── news/download_datasets.py
│   ├── wiki/crawl_vi_wiki.py
│   ├── wiki/process_vi_wiki.py
│   ├── poem/crawl_poem.py
│   ├── poem/scrape_poem_content.py
│   ├── poem/prepare_poem_data.py
│   └── deduplicate.py
├── scripts/
│   ├── train_1.sh
│   └── train_2.sh
├── artifacts/                  # Tokenizer, checkpoints, logs, final models
└── data/                       # Stage-organized datasets
```

### Data layout

```text
data/
├── stage_1/
│   ├── raw/                    # Stage-1 raw inputs (JSONL/Parquet)
│   └── dedup/                  # Stage-1 deduplicated parquets + report
└── stage_2/
    ├── raw/                    # Poem metadata CSV + processed jsonl for training
    └── dedup/                  # Stage-2 deduplicated poem parquet
```

## Training Overview

### Stage 1: Base Language Pretraining

Train GPT-2 from random initialization on mixed Vietnamese corpora such as news and Wikipedia.

### Stage 2: Domain Adaptation for Poetry

Continue pretraining the stage-1 model on a Vietnamese poem corpus to adapt the model toward 5-word quatrain generation.

## Requirements

* Python **3.11+**
* CUDA-compatible GPU
* `flash-attn` (optional; requires a compatible CUDA toolchain)
* [uv](https://github.com/astral-sh/uv) for environment and package management

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/duongtruongbinh/vietnamese-gpt2
cd vietnamese-gpt2
uv sync
uv pip install -e .
```

Run all commands from the repository root.


## Run with Docker (Trainer + REST API + Next.js UI)

### 1) Build all services

```bash
docker compose build
```

### 2) Start chat application stack (backend + UI)

```bash
docker compose up backend ui
```

- Next.js UI: `http://localhost:3000`
- FastAPI backend: `http://localhost:8000`
- Health endpoint: `http://localhost:8000/health`
- UI gọi API qua `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000` để browser truy cập đúng host.

### 3) Enter trainer container (optional for model training)

```bash
docker compose run --rm trainer
```

Inside trainer container, run pipeline commands with `uv run`, for example:

```bash
uv run python src/train_tokenizer.py
bash scripts/train_1.sh
```

> Notes:
> - `./data` and `./artifacts` are mounted into containers so outputs persist on host machine.
> - Backend will use `MODEL_PATH=/app/artifacts/model_stage2` by default (configured in `docker-compose.yml`).
> - If model files are not available yet, backend returns a mock response so UI can still be tested.
> - For GPU training, run Docker with NVIDIA Container Toolkit support (e.g. `docker compose run --rm --gpus all trainer`).

## Chat App Architecture

```text
ui (Next.js)  --->  backend (FastAPI)  --->  local HuggingFace model (artifacts/model_stage2)
```

- `backend/app/main.py` exposes `POST /api/chat` and `GET /health`.
- `ui/app/page.js` provides a simple chat interface and calls backend through `NEXT_PUBLIC_API_BASE_URL`.
- If model loading fails, backend automatically falls back to mock replies for development.

## Pipeline

### 1. Prepare raw corpora

```bash
uv run python data_prep/news/download_datasets.py
uv run python data_prep/wiki/crawl_vi_wiki.py
uv run python data_prep/wiki/process_vi_wiki.py
```

### 2. Train the tokenizer

```bash
uv run python src/train_tokenizer.py
```

### 3. Deduplicate the pretraining data

```bash
uv run python data_prep/deduplicate.py
```

### 4. Run stage 1 pretraining

```bash
bash scripts/train_1.sh
```

### 5. Prepare the poem corpus

```bash
uv run python data_prep/poem/crawl_poem.py
uv run python data_prep/poem/scrape_poem_content.py
uv run python data_prep/poem/prepare_poem_data.py
uv run python data_prep/deduplicate_poem.py
```

### 6. Run stage 2 continued pretraining

```bash
bash scripts/train_2.sh
```

## Text Generation

Generate text with the base model:

```bash
uv run python src/generate_base.py
```

Generate poem-style text with the stage-2 model:

```bash
uv run python src/generate_poem.py
```


## Configuration

All important paths and hyperparameters are managed in:

```text
src/config.py
```

This includes:

* Dataset paths
* Tokenizer directory
* Checkpoint directory
* Sequence length
* Batch size
* Learning rate
* Training budget
* Logging and runtime settings


## Outputs

Training artifacts are stored under:

```text
artifacts/
```

Typical outputs include:

* Trained tokenizer
* Intermediate checkpoints
* Final stage-1 model
* Final stage-2 model
* Training logs

---

## Notes

* Stage 1 is intended for **general Vietnamese language modeling**.
* Stage 2 is intended for **style adaptation**, not full instruction tuning.
* For best results, ensure corpus quality and deduplication are completed before training.
* A GPU is strongly recommended for both tokenizer experimentation and model training.

---

## Project Goal

This project aims to provide a simple, practical, and extensible foundation for training Vietnamese GPT-2 models from scratch and adapting them to specific text styles such as poetry