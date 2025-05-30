#!/usr/bin/env python3
import sys
import argparse
# Using AutoClasses for more flexibility with different models
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

LOG_PREFIX = "[maibloom-aicore] " # Assuming this is your script's log prefix

def log(message, level="INFO", verbose=True):
    # Assuming pypippark adds its own prefix, so this script's log should be distinct
    # If [pypippark] is from an outer wrapper, this LOG_PREFIX is fine.
    icons = {
        "INFO":    "ℹ️",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR":   "❌"
    }
    if verbose:
        # Adding a script-specific identifier if LOG_PREFIX is generic
        print(f"{LOG_PREFIX}[gpt2-community-core] {icons.get(level, '❔')} {message}")

def main():
    parser = argparse.ArgumentParser(
        description="gpt2-community-core: A simple AI assistant using a transformer model that provides a single reply."
    )
    parser.add_argument(
        "input_text",
        help="Your input prompt for the assistant."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="The name of the pre-trained model to use from Hugging Face Hub (e.g., 'gpt2', 'gpt2-medium', 'distilgpt2', 'microsoft/DialoGPT-medium'). For better factual accuracy, try larger or more specialized models."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=80, # Slightly increased for potentially more complete answers
        help="The maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Controls randomness: lower means more deterministic, higher means more random. (Recommended: 0.7-1.0)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Filters the K most likely next tokens at each step. (0 means no filtering)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling: selects the smallest set of tokens whose cumulative probability exceeds P. (0 means no filtering, 1.0 considers all)"
    )
    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Disable sampling and use greedy decoding. Overrides temperature, top_k, top_p if set."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mode for detailed logging."
    )
    args = parser.parse_args()
    verbose = args.verbose
    user_text = args.input_text

    if not user_text.strip():
        log("Input text cannot be empty.", "ERROR", True)
        sys.exit(1)

    log(f"Loading tokenizer and model for '{args.model_name}'...", "INFO", verbose)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
    except Exception as e:
        log(f"Error loading model or tokenizer '{args.model_name}': {e}", "ERROR", True)
        sys.exit(1)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            log("Tokenizer does not have a pad_token, setting it to eos_token.", "WARNING", verbose)
            tokenizer.pad_token = tokenizer.eos_token
        else:
            log("Tokenizer does not have an eos_token or pad_token. Generation might be problematic.", "ERROR", True)
            # Add a generic pad token if no eos_token, though this is less ideal
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))


    log("Configuring generation parameters...", "INFO", verbose)
    # Use GenerationConfig for managing parameters
    # Initialize GenerationConfig, ensuring eos_token_id and pad_token_id are set
    generation_config = GenerationConfig.from_pretrained(
        args.model_name,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id # Use the tokenizer's pad_token_id
    )

    generation_config.max_new_tokens = args.max_new_tokens
    
    if args.no_sample:
        generation_config.do_sample = False
        # Warn if sampling parameters are set but won't be used
        if args.temperature != parser.get_default("temperature") or \
           args.top_k != parser.get_default("top_k") or \
           args.top_p != parser.get_default("top_p"):
            log("Warning: --temperature, --top_k, --top_p are ignored when --no_sample is used.", "WARNING", verbose)
    else:
        generation_config.do_sample = True
        generation_config.temperature = args.temperature
        generation_config.top_k = args.top_k
        generation_config.top_p = args.top_p

    # early_stopping is a valid parameter in GenerationConfig.
    # For greedy/sampling, generation primarily stops when EOS is met or max_new_tokens is reached.
    # This flag is more impactful for beam search.
    generation_config.early_stopping = True

    if verbose:
        log(f"Effective generation parameters: {generation_config.to_json_string()}", "INFO", True)

    log("Tokenizing input text...", "INFO", verbose)
    encoded_input = tokenizer(user_text, return_tensors="pt", padding=True)
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]
    
    prompt_tokens_count = input_ids.shape[1]

    log("Generating text via model inference...", "INFO", verbose)
    try:
        generated_ids = model.generate(
            input_ids,
            generation_config=generation_config,
            attention_mask=attention_mask
        )
        
        reply_token_ids = generated_ids[0][prompt_tokens_count:]
        reply = tokenizer.decode(reply_token_ids, skip_special_tokens=True).strip()
    
    except Exception as e:
        log(f"Error during text generation: {e}", "ERROR", True)
        sys.exit(1)
        
    if verbose:
        log("Generation complete!", "SUCCESS", verbose)
    
    print(reply)

if __name__ == "__main__":
    main()
