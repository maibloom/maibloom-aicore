#!/usr/bin/env python3
import sys
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel

LOG_PREFIX = "[maibloom-aicore] "

def log(message, level="INFO", verbose=True):
    icons = {
        "INFO":    "ℹ️",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR":   "❌"
    }
    if verbose:
        print(f"{LOG_PREFIX}{icons.get(level, '❔')} {message}")

def main():
    parser = argparse.ArgumentParser(
        description="maibloom-aicore: A simple AI assistant using GPT2 that provides a single reply."
    )
    parser.add_argument(
        "input_text",
        help="Your input prompt for the assistant."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mode for detailed logging."
    )
    args = parser.parse_args()
    verbose = args.verbose

    user_text = args.input_text

    log("Loading base GPT2 tokenizer and LM head model...", "INFO", verbose)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Set the pad token to the EOS token to avoid attention mask warnings.
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    log("Tokenizing input text...", "INFO", verbose)
    # Pad the input so that an attention mask is created.
    encoded_input = tokenizer(user_text, return_tensors="pt", padding=True)
    input_length = encoded_input["input_ids"].shape[1]

    log("Generating text via model inference...", "INFO", verbose)
    generated_ids = model.generate(
        encoded_input["input_ids"],
        max_length=input_length + 50,  # Generate up to 50 tokens beyond the prompt.
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        do_sample=False,          # Use greedy decoding.
        early_stopping=True
    )

    full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # Slice the output to remove the original prompt.
    reply = full_output[input_length:].strip()
    
    if verbose:
        log("Generation complete!", "SUCCESS", verbose)
    print(reply)

if __name__ == "__main__":
    main()
