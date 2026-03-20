#!/usr/bin/env python3
"""Shared utility functions."""

import logging
import unicodedata
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def normalize_text(text: str) -> str:
    """Apply Unicode NFC normalization, returning empty string for None."""
    if text is None:
        return ""
    return unicodedata.normalize("NFC", text)


def format_size(num_bytes: int) -> str:
    """Format byte count as human-readable string (B/KB/MB/GB/TB)."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def configure_root_logging(level: int = logging.INFO) -> None:
    """Initialize root logging once (safe if called from multiple entry points)."""
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_gpt2_lm_head(
    model_dir: str | Path,
    *,
    dtype: torch.dtype | None = None,
    tie_weights: bool = False,
    pad_token_to_eos: bool = False,
    eval_mode: bool | None = True,
) -> tuple[GPT2LMHeadModel, GPT2TokenizerFast, str]:
    """Load GPT2LMHeadModel and tokenizer from a saved directory.

    Returns (model, tokenizer, device) where device is cuda or cpu.
    """
    model_dir = str(Path(model_dir).resolve())
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
    if pad_token_to_eos and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kw: dict = {}
    if dtype is not None:
        load_kw["dtype"] = dtype
    model = GPT2LMHeadModel.from_pretrained(model_dir, **load_kw)
    if tie_weights:
        model.tie_weights()
    if pad_token_to_eos:
        model.config.pad_token_id = tokenizer.pad_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if eval_mode is True:
        model.eval()
    elif eval_mode is False:
        model.train()

    return model, tokenizer, device
