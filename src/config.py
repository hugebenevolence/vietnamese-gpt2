#!/usr/bin/env python3
"""Centralized configuration for the Vietnamese GPT-2 pretraining project."""

# ── Dataset ──────────────────────────────────────────────────────────────────
STAGE_1_DIR = "data/stage_1"
STAGE_2_DIR = "data/stage_2"

# Stage 1 (base pretraining): raw corpora and deduplicated corpora
STAGE_1_RAW_DIR = f"{STAGE_1_DIR}/raw"
STAGE_1_DEDUP_DIR = f"{STAGE_1_DIR}/dedup"

# Stage 2 (continued pretraining): poem corpus (raw/processed) + dedup outputs
STAGE_2_RAW_DIR = f"{STAGE_2_DIR}/raw"
STAGE_2_DEDUP_DIR = f"{STAGE_2_DIR}/dedup"

RAW_DATASETS = [
    f"{STAGE_1_RAW_DIR}/bkai_train.parquet",
    f"{STAGE_1_RAW_DIR}/vi_wiki_articles_clean.parquet",
]

# Deduped sources for training. weight = how many times the source is repeated
# in the training mixture (the deduped parquets on disk stay unique).
DATASETS = [
    {"path": f"{STAGE_1_DEDUP_DIR}/bkai_train.parquet", "weight": 1},
    {"path": f"{STAGE_1_DEDUP_DIR}/vi_wiki_articles_clean.parquet", "weight": 3},
]

# ── Tokenizer training ──────────────────────────────────────────────────────
VOCAB_SIZE = 50257
MIN_FREQUENCY = 2
SPECIAL_TOKEN = "<|endoftext|>"

# ── Model ────────────────────────────────────────────────────────────────────
BASE_MODEL = "gpt2"
MAX_LENGTH = 1024

# ── Training hyperparameters ─────────────────────────────────────────────────
SEED = 42
TOKEN_BUDGET = 2_480_000_000
EVAL_SPLIT_RATIO = 0.01
PREPROCESSING_NUM_WORKERS = 30
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 16
WARMUP_RATIO = 0.1
BF16 = True
GRADIENT_CHECKPOINTING = True
DATALOADER_NUM_WORKERS = 10
WANDB_RUN_NAME_STAGE_1 = "rand-init"
WANDB_RUN_NAME_STAGE_2 = "continued-pretrain-poem"

# ── Paths ────────────────────────────────────────────────────────────────────
TOKENIZER_DIR = "./artifacts/tokenizer"
CHECKPOINT_DIR = f"./artifacts/checkpoints/{WANDB_RUN_NAME_STAGE_1}"
MODEL_DIR = f"./artifacts/checkpoints/{WANDB_RUN_NAME_STAGE_1}/final"

# ── Inference defaults ───────────────────────────────────────────────────────
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.95
REPETITION_PENALTY = 1.2

# ── Stage 2 continued pretraining (poem corpus) ─────────────────────────────
POEM_DATA_PATH = f"{STAGE_2_RAW_DIR}/poem_stanzas.jsonl"
POEM_RAW_CSV = f"{STAGE_2_RAW_DIR}/poem_dataset.csv"
POEM_CHECKPOINT_DIR = f"./artifacts/checkpoints/{WANDB_RUN_NAME_STAGE_2}"
POEM_MODEL_DIR = f"./artifacts/checkpoints/{WANDB_RUN_NAME_STAGE_2}/final"
POEM_PREFIX = "thơ:\n"
POEM_LINES_PER_STANZA = 4
POEM_WORDS_PER_LINE = 5
POEM_EPOCHS = 20
POEM_BATCH_SIZE = 32
POEM_LEARNING_RATE = 5e-5
POEM_WEIGHT_DECAY = 0.1
POEM_MAX_LENGTH = 64
