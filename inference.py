#!/usr/bin/env python3
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from config import (
    MODEL_DIR, MAX_NEW_TOKENS, TEMPERATURE, TOP_K, TOP_P,
    REPETITION_PENALTY, DO_SAMPLE, NUM_RETURN_SEQUENCES,
)
from utils import normalize_text

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer():
    print(f"Loading model from: {MODEL_DIR} ({DEVICE})")
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_DIR)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, vocab={len(tokenizer):,}")
    return model, tokenizer


def generate_text(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
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
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

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


def interactive_mode(model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast):
    print("\nInteractive mode. Type 'quit' to exit, 'config' to change params.\n")

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

            for text in generate_text(model, tokenizer, prompt, **gen_config):
                print(f"\n{text}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


def run_test_examples(model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast):
    test_prompts = [
        "Việt Nam là một đất nước",
        "Hôm nay thời tiết rất đẹp,",
        "Trong lịch sử Việt Nam,",
        "Công nghệ trí tuệ nhân tạo",
        "Hà Nội là thủ đô của",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"[{i}/{len(test_prompts)}] {prompt}")
        for text in generate_text(model, tokenizer, prompt):
            print(text)
        print()


def main():
    model, tokenizer = load_model_and_tokenizer()
    run_test_examples(model, tokenizer)

    user_input = input("Enter interactive mode? [Y/n]: ").strip().lower()
    if user_input != "n":
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
