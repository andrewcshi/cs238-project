import torch
from torch.nn import Module
from utils.logits_processor import LogitsProcessor, GreedyProcessor
import utils.printing as printing
from typing import List


@torch.no_grad()
def autoregressive_generate(
    inputs: List[int],
    model: Module,
    max_gen_len: int = 40,
    logits_processor: LogitsProcessor = GreedyProcessor(),
    eos_tokens_id: int | List[int] = None,
    pad_token_id: int = None,
    use_cache: bool = False,
    debug: bool = False,
) -> List[int]:

    if eos_tokens_id is None:
        if model.config.eos_token_id is not None:
            eos_tokens_id = [model.config.eos_token_id]
        else:
            eos_tokens_id = [model.config.vocab_size - 1]  # Fallback
    
    if isinstance(eos_tokens_id, int):
        eos_tokens_id = [eos_tokens_id]

    if pad_token_id is None:
        pad_token_id = model.config.pad_token_id if model.config.pad_token_id is not None else model.config.eos_token_id

    prompt_len = len(inputs)
    max_seq_length = getattr(model.config, 'max_position_embeddings', 1024)
    total_len = min(max_seq_length, prompt_len + max_gen_len)

    input_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=model.device)
    input_ids[0, :prompt_len] = torch.tensor(inputs, dtype=torch.long, device=model.device)

    list_tokens_id = eos_tokens_id
    stop_tokens = torch.tensor(list_tokens_id, dtype=torch.long, device=model.device).unsqueeze(1)

    generated = []

    for curr in range(prompt_len, total_len):
        o = model(input_ids[..., :curr], past_key_values=None, use_cache=use_cache)
        logits = o.logits[..., -1, :]
        probs = logits_processor(logits)
        x = logits_processor.sample(probs)
        token_id = x.item()

        if token_id >= model.config.vocab_size or token_id < 0:
            raise ValueError(f"Generated token ID {token_id} is out of bounds for the vocabulary size {model.config.vocab_size}.")

        input_ids[0, curr] = x
        generated.append(token_id)

        if token_id in list_tokens_id:
            if debug:
                printing.end_token_found(curr)
            break

    return generated


@torch.no_grad()
def beam_search_generate(
    inputs: List[int],
    model: Module,
    max_gen_len: int = 40,
    num_beams: int = 4,
    top_k: int = 3,
    min_length: float = 5.0,
    alpha: float = 1.2,
    eos_tokens_id: int | List[int] = None,
    pad_token_id: int = None,
    debug: bool = False,
    tokenizer=None,
) -> List[int]:

    def _length_penalty_fn(length, alpha, min_length):
        return ((min_length + length) / (min_length + 1)) ** alpha

    if eos_tokens_id is None:
        eos_tokens_id = [model.config.eos_token_id] if model.config.eos_token_id is not None else [model.config.vocab_size-1]
    if isinstance(eos_tokens_id, int):
        eos_tokens_id = [eos_tokens_id]

    if pad_token_id is None:
        pad_token_id = model.config.pad_token_id if model.config.pad_token_id is not None else model.config.eos_token_id

    prompt_len = len(inputs)
    max_seq_length = getattr(model.config, 'max_position_embeddings', 1024)
    assert prompt_len < max_seq_length, "Prompt length exceeds maximum sequence length."
    total_len = min(max_seq_length, prompt_len + max_gen_len)

    input_ids = torch.full((num_beams, total_len), pad_token_id, dtype=torch.long, device=model.device)
    input_ids[:, :prompt_len] = torch.tensor(inputs, dtype=torch.long, device=model.device)
    probs = torch.full((num_beams, total_len), torch.finfo(torch.float).min, dtype=torch.float, device=model.device)
    beams_probs = torch.full((num_beams,), torch.finfo(torch.float).min, dtype=torch.float, device=model.device)
    last_indexes = torch.full((num_beams,), -1, dtype=torch.long, device=model.device)

    stop_tokens = torch.tensor(eos_tokens_id, dtype=torch.long, device=model.device)

    # Prefill
    probs[:, :prompt_len] = 1.0
    beams_probs[:] = 1.0
    o = model(input_ids[:, :prompt_len])
    curr_prob = torch.nn.functional.log_softmax(o.logits[0, -1, :], dim=-1)
    top_probs, top_tokens = torch.topk(curr_prob, num_beams, dim=-1)
    input_ids[:, prompt_len] = top_tokens
    probs[:, prompt_len] = probs[:, prompt_len - 1] + top_probs
    beams_probs[:] = probs[:, prompt_len] / _length_penalty_fn(1, alpha, min_length)
    
    for curr in range(prompt_len + 1, total_len):
        o = model(input_ids[:, :curr])
        logits = o.logits[:, -1, :]
        probs_curr = torch.nn.functional.log_softmax(logits, dim=-1)
        top_probs, top_tokens = torch.topk(probs_curr, top_k, dim=-1)

        possibilities = []
        for beam in range(num_beams):
            if last_indexes[beam] != -1:
                prob_vec = probs[beam].detach().clone()
                input_vec = input_ids[beam].detach().clone()
                possibilities.append((beams_probs[beam], input_vec, prob_vec, last_indexes[beam]))
                continue
            
            for possibility in range(top_k):
                new_prob = probs[beam, curr - 1] + top_probs[beam, possibility]
                lp = _length_penalty_fn(curr - prompt_len, alpha, min_length)
                prob_vec = probs[beam].detach().clone()
                prob_vec[curr] = new_prob
                input_vec = input_ids[beam].detach().clone()
                input_vec[curr] = top_tokens[beam, possibility]
                token_id = input_vec[curr].item()

                if token_id >= model.config.vocab_size or token_id < 0:
                    continue

                last_token_idx = -1
                if torch.isin(input_vec[curr], stop_tokens) or input_vec[curr] == pad_token_id:
                    last_token_idx = curr

                already_in = False
                for p in possibilities:
                    if torch.equal(p[1], input_vec):
                        already_in = True
                        break
                if not already_in:
                    possibilities.append((new_prob / (lp if lp != 0 else 1), input_vec, prob_vec, last_token_idx))

        possibilities.sort(key=lambda x: x[0], reverse=True)
        possibilities = possibilities[:num_beams]

        if debug and tokenizer is not None:
            printing.beam_search_step(possibilities, curr, tokenizer)

        for beam in range(num_beams):
            beams_probs[beam] = possibilities[beam][0]
            input_ids[beam] = possibilities[beam][1]
            probs[beam] = possibilities[beam][2]
            last_indexes[beam] = possibilities[beam][3]

        if torch.all(last_indexes != -1):
            if debug:
                printing.end_token_found(curr)
            break

    last_indexes[last_indexes == -1] = total_len - 1

    return input_ids[0, prompt_len : last_indexes[0] + 1].tolist()
