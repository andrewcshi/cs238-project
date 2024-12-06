import torch
from torch.nn import Module
from utils.logits_processor import LogitsProcessor, GreedyProcessor
import utils.printing as printing
from typing import List, Tuple
import time


def reward_function(accepted_tokens_count, rejected_tokens, throughput):
    return accepted_tokens_count - 0.5 * rejected_tokens + 0.01 * throughput


def measure_model_time(model, input_ids, use_cache):
    start_time = time.time()
    with torch.no_grad():
        # Ensure inputs are on model's device
        input_ids = input_ids.to(model.device)
        model(input_ids=input_ids, use_cache=use_cache)
    end_time = time.time()
    return end_time - start_time


def validate_token_ids(token_ids, vocab_size, context=""):
    """
    Validates that all token IDs are within the valid range [0, vocab_size).
    Raises a ValueError if any token ID is out of bounds.
    """
    for idx, token_id in enumerate(token_ids):
        if not (0 <= token_id < vocab_size):
            raise ValueError(f"Invalid Token ID {token_id} at position {idx} in {context}.")


def safe_sample(logits_processor, logits, tokenizer, vocab_size, debug=False):
    """
    Samples a token using the logits processor and ensures the token ID is valid.
    If invalid, replaces it with the UNK token ID or 0.
    """
    sampled_token = logits_processor.sample(logits)
    token_id = sampled_token.item()
    if not (0 <= token_id < vocab_size):
        if debug:
            print(f"[safe_sample] Invalid sampled token ID: {token_id}. Replacing with UNK token.")
        token_id = tokenizer.unk_token_id if tokenizer and tokenizer.unk_token_id is not None else 0
    return torch.tensor(token_id, device=logits.device)


def sample_from_draft_model(input_ids, drafter, logits_processor, use_cache, drafter_cache, vocab_size, tokenizer=None, debug=False):
    input_ids = input_ids.to(drafter.device)
    Mq = drafter(input_ids=input_ids, past_key_values=drafter_cache, use_cache=use_cache)
    drafter_cache = Mq.past_key_values
    draft_logits = Mq.logits[..., -1, :]
    draft_probs = logits_processor(draft_logits)
    sampled_token = safe_sample(logits_processor, draft_probs, tokenizer, vocab_size, debug)

    # **Validation of Sampled Token**
    token_id = sampled_token.item()
    if token_id < 0 or token_id >= vocab_size:
        if debug:
            print(f"[sample_from_draft_model] Invalid sampled token ID after safe_sample: {token_id}. Replacing with UNK token.")
        sampled_token = torch.tensor(tokenizer.unk_token_id if tokenizer and tokenizer.unk_token_id is not None else 0, device=drafter.device)

    return sampled_token, drafter_cache, draft_probs


def verify_with_target_model(xprefix, Y_candidates, target, logits_processor, use_cache, target_cache, candidate_probs, vocab_size, tokenizer=None, debug=False):
    if not Y_candidates:
        return [], False, target_cache

    new_tokens = torch.tensor(Y_candidates, device=target.device, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        Mp = target(input_ids=new_tokens, past_key_values=target_cache, use_cache=use_cache)
        target_cache = Mp.past_key_values
        target_logits = Mp.logits[..., -len(Y_candidates):, :]
        target_probs = logits_processor(target_logits)

    accepted_tokens = []
    rejected = False

    for i, Y_i in enumerate(Y_candidates):
        p_Yi = target_probs[0, i, Y_i].item()
        q_Yi = candidate_probs[i][Y_i].item()
        acceptance_ratio = min(1.0, p_Yi / (q_Yi + 1e-12))
        rand_val = torch.rand(1).item()

        if debug:
            print(f"Token {i}: Y_i={Y_i}, p_Yi={p_Yi:.4f}, q_Yi={q_Yi:.4f}, "
                  f"acceptance_ratio={acceptance_ratio:.4f}, rand_val={rand_val:.4f}")

        if rand_val <= acceptance_ratio:
            accepted_tokens.append(Y_i)
        else:
            rejected = True
            if debug:
                print(f"Token {i}: Rejected. Sampling from target model.")
            sampled_token = sample_from_target_model(xprefix, target, logits_processor, use_cache, target_cache, vocab_size, tokenizer, debug)
            accepted_tokens.append(sampled_token)
            break

    if debug:
        print(f"Accepted tokens: {accepted_tokens}, Rejected: {rejected}")

    return accepted_tokens, rejected, target_cache


def sample_from_target_model(xprefix, target, logits_processor, use_cache, target_cache, vocab_size, tokenizer=None, debug=False):
    with torch.no_grad():
        Mp = target(input_ids=xprefix.to(target.device), past_key_values=target_cache, use_cache=use_cache)
        target_cache = Mp.past_key_values
        p_p = logits_processor(Mp.logits[..., -1, :])
        sampled_token = safe_sample(logits_processor, p_p, tokenizer, vocab_size, debug)

        # **Validation of Sampled Token**
        token_id = sampled_token.item()
        if token_id < 0 or token_id >= vocab_size:
            if debug:
                print(f"[sample_from_target_model] Invalid sampled token ID after safe_sample: {token_id}. Replacing with UNK token.")
            sampled_token = torch.tensor(tokenizer.unk_token_id if tokenizer and tokenizer.unk_token_id is not None else 0, device=target.device)

    if debug:
        print(f"Sampled token from target model: {sampled_token.item()}")
    return sampled_token.item()


def decide_action(state, c1, c2, delta, gamma, debug=False):
    xprefix, Y_candidates = state
    if len(Y_candidates) < gamma:
        if debug:
            print(f"Y_candidates length {len(Y_candidates)} < gamma {gamma}, continuing.")
        return 'continue'

    rejection_prob = estimate_rejection_probability(state)
    threshold = (c2 + delta) / (c1 + c2 + delta)

    if debug:
        print(f"Rejection probability: {rejection_prob:.3f}, Threshold: {threshold:.3f}")

    if rejection_prob >= threshold:
        if debug:
            print("Rejection probability >= threshold, stopping.")
        return 'stop'
    else:
        if debug:
            print("Rejection probability < threshold, continuing.")
        return 'continue'


def estimate_rejection_probability(state):
    xprefix, Y_candidates = state
    base_rejection_prob = 0.1
    per_token_increment = 0.05
    rejection_prob = min(base_rejection_prob + per_token_increment * len(Y_candidates), 0.9)
    return rejection_prob


def check_end_of_sequence(tokens, eos_tokens_id):
    return any(token in eos_tokens_id for token in tokens)


def speculative_generate(
    inputs: List[int],
    drafter: Module,
    target: Module,
    tokenizer=None,
    gamma: int = 5,
    logits_processor: LogitsProcessor = GreedyProcessor(),
    max_gen_len: int = 40,
    eos_tokens_id: int | List[int] = None,
    pad_token_id: int = None,
    use_cache: bool = True,
    skip_sample_adjustment: bool = False,
    first_target: bool = True,
    debug: bool = False,
    return_training_data: bool = False,
    vocab_size: int = None,  # **New Parameter for Validation**
) -> Tuple[List[int], float, dict]:

    training_info = {
        "chosen_tokens": [],
        "chosen_logits": [],
        "step_rewards": []
    }

    if eos_tokens_id is None and tokenizer is not None:
        eos_tokens_id = [tokenizer.eos_token_id]
    elif isinstance(eos_tokens_id, int):
        eos_tokens_id = [eos_tokens_id]

    if pad_token_id is None and tokenizer is not None:
        pad_token_id = tokenizer.pad_token_id

    assert len(inputs) > 0, "Input sequence is empty."
    assert max_gen_len > 0, "max_gen_len must be positive."

    drafter_cache, target_cache = None, None

    if eos_tokens_id is None:
        eos_tokens_id = [tokenizer.eos_token_id]

    stop_tokens = torch.tensor(eos_tokens_id, dtype=torch.long, device=target.device).unsqueeze(1)
    drafts_accepted, drafts_speculated = 0, 0

    prompt_len = len(inputs)
    max_seq_length = getattr(target.config, 'max_position_embeddings', 1024)
    total_len = min(max_seq_length, prompt_len + max_gen_len)
    assert total_len > prompt_len, \
        f"Total length ({total_len}) must be greater than prompt length ({prompt_len})."

    input_ids = torch.full(
        (1, total_len),
        pad_token_id,
        dtype=torch.long,
        device=target.device,
        requires_grad=False
    )
    input_ids[0, :prompt_len] = torch.tensor(inputs, dtype=torch.long, device=target.device)

    current_position = prompt_len

    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)

    with torch.no_grad():
        sample_input_ids = input_ids[..., :current_position]
        tdraft = measure_model_time(drafter, sample_input_ids, use_cache)
        ttarget = measure_model_time(target, sample_input_ids, use_cache)
    c1 = tdraft
    c2 = max(ttarget - tdraft, 1e-6)
    delta = 1e-6

    xprefix = input_ids[..., :current_position]
    Y_candidates = []
    Y_candidate_probs = []
    steps = 0
    accepted_tokens_total = 0
    cumulative_reward = 0.0

    if first_target:
        with torch.no_grad():
            Mp = target(input_ids=input_ids[..., :current_position], past_key_values=target_cache, use_cache=use_cache)
            target_cache = Mp.past_key_values
            p_p = logits_processor(Mp.logits[..., -1, :])
            t = safe_sample(logits_processor, p_p, tokenizer, vocab_size, debug)

            # **Validation of Sampled Token**
            validate_token_ids([t.item()], vocab_size, context="Initial Target Sampled Token")

        if current_position < total_len:
            with torch.no_grad():
                input_ids[0, current_position] = t
            current_position += 1
        else:
            return input_ids[0, prompt_len:current_position].tolist(), 1.0, training_info

        xprefix = input_ids[..., :current_position]
        if torch.isin(t, stop_tokens):
            return input_ids[0, prompt_len:current_position].tolist(), 1.0, training_info
        if debug:
            printing.initial_step(t, tokenizer)

    max_iterations = max_gen_len * 2
    iteration = 0
    start_generation_time = time.time()

    while current_position < total_len and iteration < max_iterations:
        iteration += 1
        steps += 1
        action = decide_action((xprefix, Y_candidates), c1, c2, delta, gamma, debug)

        if action == 'continue' and len(Y_candidates) < gamma:
            try:
                sampled_token, drafter_cache, draft_probs = sample_from_draft_model(
                    xprefix, drafter, logits_processor, use_cache, drafter_cache, vocab_size, tokenizer, debug
                )
                # **Move sampled_token to target device**
                sampled_token = sampled_token.to(input_ids.device)
                # **Validate sampled token**
                validate_token_ids([sampled_token.item()], vocab_size, context="Draft Sampled Token")
            except Exception as e:
                if debug:
                    print(f"[Draft Model Error] {e}")
                break

            if return_training_data:
                training_info["chosen_tokens"].append(int(sampled_token.item()))
                training_info["chosen_logits"].append(draft_probs.squeeze(0))

            Y_candidates.append(int(sampled_token.item()))
            Y_candidate_probs.append(draft_probs.squeeze(0))

            if current_position < total_len:
                with torch.no_grad():
                    input_ids[0, current_position] = sampled_token
                current_position += 1
                xprefix = input_ids[..., :current_position]
                drafts_speculated += 1
            else:
                if debug:
                    print(f"Warning: Reached total_len={total_len}. Cannot assign more tokens.")
                break
        else:
            if len(Y_candidates) == 0:
                continue
            try:
                accepted_tokens, rejected, target_cache = verify_with_target_model(
                    xprefix, Y_candidates, target, logits_processor, use_cache, target_cache, Y_candidate_probs, vocab_size, tokenizer, debug
                )
                # Validate accepted tokens
                validate_token_ids(accepted_tokens, vocab_size, context="Accepted Tokens")
            except Exception as e:
                if debug:
                    print(f"[Verification Error] {e}")
                break

            drafts_accepted += len(accepted_tokens)
            accepted_tokens_total += len(accepted_tokens)
            rejected_count = 1 if rejected else 0

            if accepted_tokens:
                for token in accepted_tokens:
                    if current_position < total_len:
                        with torch.no_grad():
                            input_ids[0, current_position] = token
                        current_position += 1
                        xprefix = input_ids[..., :current_position]
                    else:
                        if debug:
                            print(f"Warning: Reached total_len={total_len}.")
                        break
            else:
                try:
                    sampled_token = sample_from_target_model(xprefix, target, logits_processor, use_cache, target_cache, vocab_size, tokenizer, debug)
                    # **Validation of Sampled Token**
                    validate_token_ids([sampled_token], vocab_size, context="Target Sampled Token")
                except Exception as e:
                    if debug:
                        print(f"[Target Model Sampling Error] {e}")
                    break

                if current_position < total_len:
                    with torch.no_grad():
                        input_ids[0, current_position] = sampled_token
                    current_position += 1
                    xprefix = input_ids[..., :current_position]
                    accepted_tokens = [sampled_token]
                else:
                    if debug:
                        print(f"Warning: Reached total_len={total_len}.")
                    break

            elapsed_time = time.time() - start_generation_time
            total_generated = current_position - prompt_len
            throughput = total_generated / max(1e-6, elapsed_time)
            step_reward = reward_function(len(accepted_tokens), rejected_count, throughput)
            cumulative_reward += step_reward

            if return_training_data:
                training_info["step_rewards"].append(step_reward)

            if check_end_of_sequence(accepted_tokens, eos_tokens_id):
                break

            Y_candidates = []
            Y_candidate_probs = []

    generated_tokens = input_ids[0, prompt_len:current_position].tolist()
    acceptance_rate = drafts_accepted / drafts_speculated if drafts_speculated > 0 else 1.0

    return generated_tokens, acceptance_rate, training_info
