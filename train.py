import argparse
import torch
import torch.optim as optim
import time
import random
import numpy as np
import gc
import os
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.logits_processor import GreedyProcessor
from sampling.speculative_decoding import speculative_generate
from sampling.base_decoding import autoregressive_generate
from datasets import load_dataset

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validate_token_ids(token_ids, vocab_size, context=""):
    """
    Validates that all token IDs are within the valid range [0, vocab_size).
    Raises a ValueError if any token ID is out of bounds.
    """
    for idx, token_id in enumerate(token_ids):
        if not (0 <= token_id < vocab_size):
            raise ValueError(f"Invalid Token ID {token_id} at position {idx} in {context}.")

def main():
    parser = argparse.ArgumentParser(description="Speculative Decoding Train & Test")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training and inference")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    
    # Correctly configure mutually exclusive group to set the same 'finetune' attribute
    finetune_group = parser.add_mutually_exclusive_group()
    finetune_group.add_argument("--finetune", dest='finetune', action="store_true", help="Enable finetuning (default)")
    finetune_group.add_argument("--no_finetune", dest='finetune', action="store_false", help="Disable finetuning")
    parser.set_defaults(finetune=True)

    parser.add_argument(
        "--hf_dataset_train",
        type=str,
        default=None,
        help="Name of the HuggingFace dataset to use for training (e.g., 'wikitext')."
    )
    parser.add_argument(
        "--hf_config_train",
        type=str,
        default=None,
        help="Configuration name of the HuggingFace dataset (if applicable)."
    )
    parser.add_argument(
        "--hf_split_train",
        type=str,
        default="train",
        help="Split of the HuggingFace dataset to use for training (e.g., 'train')."
    )
    parser.add_argument(
        "--hf_field_train",
        type=str,
        default="text",
        help="Field name in the HuggingFace dataset that contains the text data."
    )

    parser.add_argument(
        "--hf_dataset_test",
        type=str,
        default=None,
        help="Name of the HuggingFace dataset to use for testing (e.g., 'wikitext')."
    )
    parser.add_argument(
        "--hf_config_test",
        type=str,
        default=None,
        help="Configuration name of the HuggingFace dataset (if applicable)."
    )
    parser.add_argument(
        "--hf_split_test",
        type=str,
        default="test",
        help="Split of the HuggingFace dataset to use for testing (e.g., 'train', 'validation', 'test')."
    )
    parser.add_argument(
        "--hf_field_test",
        type=str,
        default="text",
        help="Field name in the HuggingFace dataset that contains the text data."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for verbose logging."
    )
    
    args = parser.parse_args()

    # Set environment variable for memory fragmentation before any CUDA operations
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    set_seed(42)

    device = args.device
    use_mixed_precision = args.mixed_precision
    finetune = args.finetune
    debug = args.debug

    # Models
    # Switch to smaller models to mitigate CUDA OOM
    target_model = "gpt2-xl"
    drafter_model = "gpt2"

    print("Loading target model:", target_model)
    print("Loading drafter model:", drafter_model)

    tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    # Load target and drafter models
    target = AutoModelForCausalLM.from_pretrained(
        target_model,
        device_map='cpu',
        trust_remote_code=True
    )
    target.eval()

    drafter = AutoModelForCausalLM.from_pretrained(
        drafter_model,
        device_map=device,
        trust_remote_code=True
    )
    drafter.train()

    # Enable gradient checkpointing to save memory
    drafter.gradient_checkpointing_enable()

    target_vocab_size = target.config.vocab_size
    drafter_vocab_size = drafter.config.vocab_size

    if vocab_size != target_vocab_size:
        raise ValueError(f"Tokenizer vocab size ({vocab_size}) does not match target model vocab size ({target_vocab_size}).")
    if vocab_size != drafter_vocab_size:
        raise ValueError(f"Tokenizer vocab size ({vocab_size}) does not match drafter model vocab size ({drafter_vocab_size}).")

    # Existing hardcoded dataset for finetuning and default testing
    default_dataset = [
        "Once upon a time",
        "In a distant galaxy",
        "The scientist discovered a new element",
        "Under the starry sky, a lone wolf howled",
        "The president addressed the nation",
        "A new species of bird was found",
        "In the quantum world, particles behave strangely",
        "A young boy dreamed of adventures",
        "In the deep sea, creatures glow",
        "The ancient ruins held many secrets"
    ]

    finetune_data = default_dataset[:int(len(default_dataset) * 0.2)]
    test_data = default_dataset[int(len(default_dataset) * 0.2):]

    if args.hf_dataset_train:
        try:
            print(f"[train] Loading HuggingFace dataset: {args.hf_dataset_train}")
            dataset = load_dataset(args.hf_dataset_train, args.hf_config_train, split=args.hf_split_train)
            finetune_data = dataset[args.hf_field_train]
            print(f"[train] Loaded {len(finetune_data)} examples from '{args.hf_dataset_train}' dataset.")
        except Exception as e:
            print(f"[train] Error loading HuggingFace dataset: {e}")
            print("[train] Falling back to the default hardcoded train dataset.")

    if args.hf_dataset_test:
        try:
            print(f"[test] Loading HuggingFace dataset: {args.hf_dataset_test}")
            dataset = load_dataset(args.hf_dataset_test, args.hf_config_test, split=args.hf_split_test)
            test_data = dataset[args.hf_field_test]
            print(f"[test] Loaded {len(test_data)} examples from '{args.hf_dataset_test}' dataset.")
        except Exception as e:
            print(f"[test] Error loading HuggingFace dataset: {e}")
            print("[test] Falling back to the default hardcoded test dataset.")

    optimizer = optim.AdamW(drafter.parameters(), lr=1e-5)
    processor = GreedyProcessor()
    eos_tokens_id = [tokenizer.eos_token_id]

    scaler = torch.amp.GradScaler(enabled=use_mixed_precision)

    # If we are finetuning with gradient checkpointing, set use_cache=False explicitly
    # because they are incompatible. This will be used in calls to speculative_generate
    # and autoregressive_generate.
    training_use_cache = False if finetune else True

    if finetune:
        epochs = 2
        print("Starting Finetuning...")
        drafter.train()
        for epoch in tqdm(range(epochs), desc="Finetuning Epochs"):
            total_reward = 0.0
            total_loss = 0.0
            count = 0

            for prefix in tqdm(finetune_data, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                tokenized = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
                tokenized = tokenized.squeeze(0).tolist()
                try:
                    output_ids, accept_rate, training_info = speculative_generate(
                        tokenized,
                        drafter,
                        target,  
                        tokenizer=tokenizer,
                        logits_processor=processor,
                        gamma=5,
                        max_gen_len=20,
                        eos_tokens_id=eos_tokens_id,
                        debug=debug,
                        use_cache=training_use_cache,  # Correctly set based on finetune flag
                        return_training_data=True,
                        vocab_size=vocab_size
                    )
                except Exception as e:
                    print(f"[Speculative Generate Error] {e}")
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue

                # Validate output
                try:
                    validate_token_ids(output_ids, vocab_size, context="Speculative Generate Output")
                except ValueError as ve:
                    print(f"[Validation Error] {ve}")
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue

                step_rewards = training_info.get("step_rewards", [])
                chosen_tokens = training_info.get("chosen_tokens", [])
                chosen_logits_list = training_info.get("chosen_logits", [])

                rollout_reward = sum(step_rewards)
                total_reward += rollout_reward

                if len(chosen_tokens) > 0:
                    log_probs = []
                    invalid_token_found = False
                    for logits, token_id in zip(chosen_logits_list, chosen_tokens):
                        if logits.device != device:
                            logits = logits.to(device)
                        lprobs = torch.log_softmax(logits, dim=-1)
                        if token_id >= lprobs.shape[-1] or token_id < 0:
                            print("[Log Prob Error] Token ID out of range, skipping this example.")
                            invalid_token_found = True
                            break
                        log_prob = lprobs[token_id]
                        log_probs.append(log_prob)

                    if invalid_token_found or len(log_probs) == 0:
                        # Skip this example entirely
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue

                    log_probs = torch.stack(log_probs)
                    if not log_probs.requires_grad:
                        print("Error: log_probs do not require gradients.")
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue

                    loss = -(rollout_reward * log_probs.mean())
                    total_loss += loss.item()

                    optimizer.zero_grad(set_to_none=True)
                    if use_mixed_precision:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    del log_probs, loss, output_ids, training_info
                    torch.cuda.empty_cache()
                    gc.collect()

                count += 1

            avg_reward = total_reward / max(1, count)
            avg_loss = total_loss / max(1, count)
            tqdm.write(f"Epoch {epoch+1}/{epochs} - Avg Reward: {avg_reward:.4f}, Avg Loss: {avg_loss:.4f}")
            gc.collect()
            torch.cuda.empty_cache()

        drafter.eval()
        print("Finetuning complete!")
    else:
        print("Finetuning is disabled. Skipping finetuning phase.")

    # Evaluate on test data
    print("Evaluating on test data...")
    drafter.eval()
    total_acceptance = 0.0
    total_spec_throughput = 0.0
    total_base_throughput = 0.0
    test_count = 0

    for prefix in tqdm(test_data, desc="Testing"):
        if isinstance(prefix, dict):
            prefix = prefix.get(args.hf_field_test, "")
        elif not isinstance(prefix, str):
            prefix = str(prefix)
        
        if not prefix.strip():
            if debug:
                print("Skipping empty prefix.")
            continue

        tokenized = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
        tokenized = tokenized.squeeze(0).tolist()

        if not tokenized:
            if debug:
                print(f"Skipping prefix due to empty tokenization: '{prefix}'")
            continue

        # Speculative Decoding
        try:
            spec_start_time = time.time()
            spec_output_ids, spec_accept_rate, _ = speculative_generate(
                tokenized,
                drafter,
                target,
                tokenizer=tokenizer,
                logits_processor=processor,
                gamma=5,
                max_gen_len=20,
                eos_tokens_id=eos_tokens_id,
                debug=debug,
                use_cache=False,  # Use_cache disabled during eval to avoid issues
                return_training_data=False,
                vocab_size=vocab_size
            )
            spec_end_time = time.time()
        except Exception as e:
            print(f"[Speculative Generate Error] {e}")
            gc.collect()
            torch.cuda.empty_cache()
            continue

        try:
            validate_token_ids(spec_output_ids, vocab_size, context="Speculative Generate Evaluation Output")
        except ValueError as ve:
            print(f"[Speculative Output Validation Error] {ve}")
            gc.collect()
            torch.cuda.empty_cache()
            continue

        spec_output_text = tokenizer.decode(spec_output_ids, skip_special_tokens=True)
        spec_throughput = len(spec_output_text) / max(1e-6, (spec_end_time - spec_start_time))

        # Autoregressive Decoding (Baseline)
        try:
            base_start_time = time.time()
            base_output_ids = autoregressive_generate(
                tokenized,
                target,
                use_cache=False,  # Also disable here for consistency
                max_gen_len=20,
                eos_tokens_id=eos_tokens_id,
                logits_processor=processor,
                debug=debug,
            )
            base_end_time = time.time()
        except Exception as e:
            print(f"[Autoregressive Generate Error] {e}")
            gc.collect()
            torch.cuda.empty_cache()
            continue

        try:
            validate_token_ids(base_output_ids, vocab_size, context="Autoregressive Generate Evaluation Output")
        except ValueError as ve:
            print(f"[Baseline Output Validation Error] {ve}")
            gc.collect()
            torch.cuda.empty_cache()
            continue

        base_output_text = tokenizer.decode(base_output_ids, skip_special_tokens=True)
        base_throughput = len(base_output_text) / max(1e-6, (base_end_time - base_start_time))

        if base_throughput > 0:
            throughput_increase = ((spec_throughput / base_throughput) - 1) * 100
        else:
            throughput_increase = 0.0

        total_acceptance += spec_accept_rate
        total_spec_throughput += spec_throughput
        total_base_throughput += base_throughput
        test_count += 1

        print(f"Prompt: {prefix}")
        print(f"Speculative Output: \n\n{spec_output_text}\n")
        print(f"Autoregressive Output: \n\n{base_output_text}\n")
        print(f"Acceptance Rate: {spec_accept_rate:.3f}")
        print(f"Speculative Throughput: {spec_throughput:.1f} tokens/s")
        print(f"Baseline Throughput: {base_throughput:.1f} tokens/s")
        print(f"Throughput Increase: {throughput_increase:.1f}%\n")

        del spec_output_ids, spec_output_text, base_output_ids, base_output_text
        torch.cuda.empty_cache()
        gc.collect()

    avg_acceptance = total_acceptance / test_count if test_count > 0 else 0.0
    avg_spec_throughput = total_spec_throughput / test_count if test_count > 0 else 0.0
    avg_base_throughput = total_base_throughput / test_count if test_count > 0 else 0.0

    if avg_base_throughput > 0:
        avg_throughput_increase = ((avg_spec_throughput / avg_base_throughput) - 1) * 100
    else:
        avg_throughput_increase = 0.0

    print("Test Results:")
    print(f"Avg Acceptance Rate: {avg_acceptance:.3f}")
    print(f"Avg Speculative Throughput: {avg_spec_throughput:.1f} tokens/s")
    print(f"Avg Baseline Throughput: {avg_base_throughput:.1f} tokens/s")
    print(f"Avg Throughput Increase: {avg_throughput_increase:.1f}%")

if __name__ == "__main__":
    main()
