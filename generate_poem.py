#!/usr/bin/env python3
"""Generate 5-word quatrains from SFT model."""

import logging

import torch

from config import (
    POEM_MODEL_DIR, POEM_PREFIX,
    TEMPERATURE, TOP_K, TOP_P, REPETITION_PENALTY,
)
from utils import configure_root_logging, load_gpt2_lm_head

logger = logging.getLogger(__name__)


def load_model():
    model, tokenizer, device = load_gpt2_lm_head(
        POEM_MODEL_DIR,
        dtype=torch.bfloat16,
        tie_weights=True,
        pad_token_to_eos=True,
        eval_mode=True,
    )
    return tokenizer, model, device


def generate(tokenizer, model, device, prompt="", num_samples=2):
    """Generate poem from prompt (can be empty or first line)."""
    text = POEM_PREFIX + prompt
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            num_return_sequences=num_samples,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    results = []
    for out in outputs:
        text = tokenizer.decode(out, skip_special_tokens=True)
        if text.startswith("thơ:"):
            text = text[4:].strip()
        results.append(text)
    return results


def main() -> None:
    configure_root_logging()
    tokenizer, model, device = load_model()

    prompts = [
        "",
        "Trăng sáng trên đầu núi",
        "Mùa thu lá vàng rơi",
        "Hoa sen nở trắng hồ",
    ]

    for prompt in prompts:
        logger.info("\n[Prompt: %s]", prompt or "(empty)")
        poems = generate(tokenizer, model, device, prompt)
        for poem in poems:
            logger.info("%s", poem)
            logger.info("")

    logger.info("Interactive mode (type 'q' to quit)")
    while True:
        prompt = input("\nFirst line: ").strip()
        if prompt.lower() == "q":
            break

        poems = generate(tokenizer, model, device, prompt)
        for poem in poems:
            logger.info("%s", poem)
            logger.info("")


if __name__ == "__main__":
    main()
