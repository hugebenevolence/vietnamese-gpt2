# data_prep/deduplicate_poem.py
#!/usr/bin/env python3
import json
import hashlib
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from loguru import logger
from transformers import GPT2TokenizerFast

from src import config as cfg
from src.utils import normalize_text

DEDUP_DIR = Path(cfg.STAGE_2_DEDUP_DIR)
MIN_DOC_CHARS = 20
BATCH_SIZE = 50_000
TOKEN_BATCH_SIZE = 8192

def sha_bytes(text: str) -> bytes:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).digest()

def iter_poem_texts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)["text"]

def flush_rows(writer, out_path, rows):
    if not rows:
        return writer
    table = pa.table({"text": rows})
    if writer is None:
        writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
    writer.write_table(table)
    rows.clear()
    return writer

def dedup_poem():
    os.makedirs(DEDUP_DIR, exist_ok=True)

    in_path = Path(cfg.POEM_DATA_PATH)
    out_path = DEDUP_DIR / f"{in_path.stem}.parquet"

    seen = set()
    writer = None
    out_rows = []

    original_docs = 0
    deduped_docs = 0

    for text in iter_poem_texts(str(in_path)):
        original_docs += 1
        text = (text or "").strip()
        if len(text) < MIN_DOC_CHARS:
            continue

        h = sha_bytes(text)
        if h in seen:
            continue
        seen.add(h)

        out_rows.append(normalize_text(text))
        deduped_docs += 1

        if len(out_rows) >= BATCH_SIZE:
            writer = flush_rows(writer, out_path, out_rows)

    writer = flush_rows(writer, out_path, out_rows)
    if writer is not None:
        writer.close()

    logger.info("Poem dedup: {} -> {}", f"{original_docs:,}", f"{deduped_docs:,}")
    logger.info("Saved to {}", out_path)
    return out_path

def count_stage2_tokens(dedup_parquet_path: str):
    tokenizer = GPT2TokenizerFast.from_pretrained(cfg.TOKENIZER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("parquet", data_files=dedup_parquet_path, split="train")
    ds = ds.train_test_split(test_size=0.1, seed=cfg.SEED)

    eos = tokenizer.eos_token or ""
    prefix = cfg.POEM_PREFIX

    def count_split(split_ds):
        total = 0
        for i in range(0, len(split_ds), TOKEN_BATCH_SIZE):
            texts = [
                prefix + normalize_text(t) + eos
                for t in split_ds[i:i + TOKEN_BATCH_SIZE]["text"]
            ]
            enc = tokenizer(
                texts,
                truncation=True,
                max_length=cfg.POEM_MAX_LENGTH,
                padding=False,
                add_special_tokens=False,
                return_attention_mask=False,
                return_length=True,
            )
            total += sum(enc["length"])
        return total

    train_tokens = count_split(ds["train"])
    eval_tokens = count_split(ds["test"])

    logger.info("Train samples: {}", len(ds["train"]))
    logger.info("Eval samples : {}", len(ds["test"]))
    logger.info("Train tokens : {}", f"{train_tokens:,}")
    logger.info("Eval tokens  : {}", f"{eval_tokens:,}")
    logger.info("Effective train tokens over {} epochs: {}", cfg.POEM_EPOCHS, f"{train_tokens * cfg.POEM_EPOCHS:,}")

def main():
    dedup_path = dedup_poem()
    count_stage2_tokens(str(dedup_path))

if __name__ == "__main__":
    main()