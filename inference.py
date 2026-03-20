#!/usr/bin/env python3
import logging

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from config import (
    MODEL_DIR, MAX_NEW_TOKENS, TEMPERATURE, TOP_K, TOP_P,
    REPETITION_PENALTY, DO_SAMPLE, NUM_RETURN_SEQUENCES,
)
from utils import configure_root_logging, load_gpt2_lm_head, normalize_text

logger = logging.getLogger(__name__)


def load_model_and_tokenizer() -> tuple[GPT2LMHeadModel, GPT2TokenizerFast, str]:
    model, tokenizer, device = load_gpt2_lm_head(MODEL_DIR, eval_mode=True)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(
        "Loaded model from %s (%s) — %.1fM params, vocab=%s",
        MODEL_DIR, device, n_params, f"{len(tokenizer):,}",
    )
    return model, tokenizer, device


def generate_text(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    device: str,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    top_p: float = TOP_P,
    repetition_penalty: float = REPETITION_PENALTY,
    do_sample: bool = DO_SAMPLE,
    num_return_sequences: int = NUM_RETURN_SEQUENCES,
) -> list[str]:
    prompt = normalize_text(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


def interactive_mode(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    device: str,
) -> None:
    logger.info("Interactive mode. Type 'quit' to exit, 'config' to change params.")

    gen_config = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
    }

    while True:
        try:
            prompt = input("Prompt: ").strip()

            if prompt.lower() in ["quit", "exit", "q"]:
                break

            if prompt.lower() == "config":
                for key, value in gen_config.items():
                    new_value = input(f"  {key} [{value}]: ").strip()
                    if new_value:
                        gen_config[key] = type(value)(new_value)
                continue

            if not prompt:
                continue

            for text in generate_text(model, tokenizer, device, prompt, **gen_config):
                logger.info("\n%s\n", text)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.exception("Generation error: %s", e)


def run_test_examples(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    device: str,
) -> None:
    test_prompts = [
        "Việt Nam là một đất nước",
        "Hôm nay thời tiết rất đẹp,",
        "Trong lịch sử Việt Nam,",
        "Công nghệ trí tuệ nhân tạo",
        "Hà Nội là thủ đô của",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        logger.info("[%d/%d] %s", i, len(test_prompts), prompt)
        for text in generate_text(model, tokenizer, device, prompt):
            logger.info("%s", text)
        logger.info("")


def main() -> None:
    configure_root_logging()
    model, tokenizer, device = load_model_and_tokenizer()
    run_test_examples(model, tokenizer, device)

    user_input = input("Enter interactive mode? [Y/n]: ").strip().lower()
    if user_input != "n":
        interactive_mode(model, tokenizer, device)


if __name__ == "__main__":
    main()
