#!/usr/bin/env python3
import logging

from src.config import (
    MODEL_DIR, MAX_NEW_TOKENS, TEMPERATURE,
    TOP_K, TOP_P, REPETITION_PENALTY,
)
from src.utils import configure_root_logging, generate_texts, load_gpt2

logger = logging.getLogger(__name__)

DEFAULT_GEN_CONFIG = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "temperature": TEMPERATURE,
    "top_k": TOP_K,
    "top_p": TOP_P,
    "repetition_penalty": REPETITION_PENALTY,
    "do_sample": True,
}


def interactive_mode(model, tokenizer, device):
    logger.info("Interactive mode. Type 'quit' to exit, 'config' to change params.")
    gen_config = dict(DEFAULT_GEN_CONFIG)

    while True:
        try:
            prompt = input("Prompt: ").strip()

            if prompt.lower() in ("quit", "exit", "q"):
                break
            if prompt.lower() == "config":
                for key, value in gen_config.items():
                    new_value = input(f"  {key} [{value}]: ").strip()
                    if new_value:
                        gen_config[key] = type(value)(new_value)
                continue
            if not prompt:
                continue

            for text in generate_texts(model, tokenizer, device, prompt, **gen_config):
                logger.info("\n%s\n", text)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.exception("Generation error: %s", e)


def main():
    configure_root_logging()
    model, tokenizer, device = load_gpt2(MODEL_DIR, eval_mode=True)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(
        "Loaded model from %s (%s) — %.1fM params, vocab=%s",
        MODEL_DIR, device, n_params, f"{len(tokenizer):,}",
    )

    test_prompts = [
        "Việt Nam là một đất nước",
        "Hôm nay thời tiết rất đẹp,",
        "Trong lịch sử Việt Nam,",
        "Công nghệ trí tuệ nhân tạo",
        "Hà Nội là thủ đô của",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        logger.info("[%d/%d] %s", i, len(test_prompts), prompt)
        for text in generate_texts(model, tokenizer, device, prompt, **DEFAULT_GEN_CONFIG):
            logger.info("%s", text)
        logger.info("")

    user_input = input("Enter interactive mode? [Y/n]: ").strip().lower()
    if user_input != "n":
        interactive_mode(model, tokenizer, device)


if __name__ == "__main__":
    main()
