"""
Day 1, Part 3 -- Turning scores into a choice.
The sample() function, plus exercises 11-13.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()


def sample(logits, temperature, top_k, top_p):
    """Return one token id, given the model's logits for a whole sequence.

    temperature: divide logits by this before softmax. T -> 0 is greedy.
    top_k: keep only the k highest-probability tokens before sampling.
    top_p (nucleus): keep the smallest set of tokens whose cumulative
                     probability exceeds p.
    """
    next_logits = logits[:, -1, :]  # only the LAST position matters

    # Hint: T=0 must be a hard argmax short-circuit, never a division --
    # dividing by 0 crashes, and dividing by something tiny just makes an
    # extremely spiky softmax, which is not the same thing as true greedy.
    if temperature == 0:
        return torch.argmax(next_logits, dim=-1, keepdim=True)

    probs = torch.softmax(next_logits / temperature, dim=-1)

    # keep the top_k highest, everything else gets discarded
    topk_probs, topk_indices = torch.topk(probs, top_k)

    # nucleus (top-p): keep the smallest prefix whose cumulative
    # probability exceeds top_p
    cum_probs = torch.cumsum(topk_probs, dim=-1)
    nucleus = cum_probs <= top_p
    nucleus[:, 0] = True  # always keep at least the single best candidate
    topk_probs = topk_probs.clone()
    topk_probs[~nucleus] = 0

    # sample a RANK (a position 0..k-1 inside the topk slice) ...
    rank = torch.multinomial(topk_probs, num_samples=1)
    # ... then translate that rank back into a REAL vocabulary id.
    # This gather is not optional: without it you get the rank itself
    # (a number 0..k-1) instead of the actual token id.
    next_token_id = topk_indices.gather(-1, rank)
    return next_token_id


def quick_generate(prompt, num_new_tokens, temperature, top_k, top_p):
    """A minimal loop just for exercises 11-13 below. The real naive/cached
    loops with EOS handling and timing live in generate.py."""
    ids = torch.tensor([tokenizer.encode(prompt)])
    for _ in range(num_new_tokens):
        with torch.no_grad():
            logits = model(ids).logits
        next_id = sample(logits, temperature, top_k, top_p)
        ids = torch.cat([ids, next_id], dim=-1)
    return tokenizer.decode(ids[0])


# ---------------------------------------------------------------
# Exercise 11: temperature sweep
# ---------------------------------------------------------------
prompt = "The future of AI"
for T in [0.1, 0.7, 1.0, 1.5, 2.0]:
    torch.manual_seed(0)  # same seed -> differences come from T, not luck
    text = quick_generate(prompt, num_new_tokens=50, temperature=T, top_k=50, top_p=1.0)
    print(f"\n--- T={T} ---\n{text}")



# ---------------------------------------------------------------
# Exercise 12: find a loop, then the smallest T that breaks it
# ---------------------------------------------------------------
loop_prone_prompt = "The the the"  # repetitive prompts loop easily -- try a few
torch.manual_seed(0)
print(quick_generate(loop_prone_prompt, 40, temperature=0, top_k=1, top_p=1.0))
# greedy (T=0) is deterministic and prone to repeating itself: once it's
# in a loop, argmax keeps picking the same continuation forever.

for T in [0.1, 0.3, 0.5, 0.7]:
    torch.manual_seed(0)
    text = quick_generate(loop_prone_prompt, 40, temperature=T, top_k=50, top_p=1.0)
    print(f"\nT={T}: {text}")
