#!/usr/bin/env python3
"""Generate 5-word quatrains from SFT model."""

from loguru import logger

import torch

from src.config import (
    POEM_MODEL_DIR, POEM_PREFIX,
    TEMPERATURE, TOP_K, TOP_P, REPETITION_PENALTY,
)
from src.utils import generate_texts, load_gpt2

def generate_poems(tokenizer, model, device, prompt: str = "", num_samples: int = 2) -> list[str]:
    """Generate poems from optional first-line prompt; strip SFT prefix from output."""
    full_prompt = POEM_PREFIX + prompt
    texts = generate_texts(
        model,
        tokenizer,
        device,
        full_prompt,
        max_new_tokens=60,
        num_return_sequences=num_samples,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
    )
    return [t[4:].strip() if t.startswith("thơ:") else t for t in texts]

def main():
    model, tokenizer, device = load_gpt2(
        POEM_MODEL_DIR,
        torch_dtype=torch.bfloat16,
        tie_weights=True,
        pad_token_to_eos=True,
        eval_mode=True,
    )

    prompts = [
        "",
        "Trăng sáng trên đầu núi",
        "Mùa thu lá vàng rơi",
        "Hoa sen nở trắng hồ",
    ]

    for prompt in prompts:
        logger.info("\n[Prompt: {}]", prompt or "(empty)")
        for poem in generate_poems(model, tokenizer, device, prompt):
            logger.info("{}\n", poem)

    logger.info("Interactive mode (type 'q' to quit)")
    while True:
        prompt = input("\nFirst line: ").strip()
        if prompt.lower() == "q":
            break
        for poem in generate_poems(model, tokenizer, device, prompt):
            logger.info("{}\n", poem)

if __name__ == "__main__":
    main()
