"""
Day 1, Parts 4 & 5 -- The generation loop, naive and cached.
Exercises 14-23.
"""

import time
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from sampling import sample
import warnings
warnings.filterwarnings('ignore')

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

EOS_ID = tokenizer.eos_token_id  # exercise 15: GPT-2's end-of-sequence id


# ---------------------------------------------------------------
# Exercise 14 + 15: the naive loop, with an EOS stopping condition
# ---------------------------------------------------------------
def generate_naive(prompt, num_new_tokens, temperature, top_k, top_p):
    ids = torch.tensor([tokenizer.encode(prompt)])
    for _ in range(num_new_tokens):
        with torch.no_grad():
            logits = model(ids).logits
        next_id = sample(logits, temperature, top_k, top_p)
        ids = torch.cat([ids, next_id], dim=-1)
        if next_id.item() == EOS_ID:  # exercise 15
            break
    return tokenizer.decode(ids[0])


# ---------------------------------------------------------------
# Exercises 16 + 17: time it, then time PER TOKEN vs token index
# ---------------------------------------------------------------
def time_naive_per_token(prompt, num_new_tokens, temperature, top_k, top_p):
    ids = torch.tensor([tokenizer.encode(prompt)])
    times = []
    for _ in range(num_new_tokens):
        start = time.process_time()  # excludes OS-scheduling noise
        with torch.no_grad():
            logits = model(ids).logits
        next_id = sample(logits, temperature, top_k, top_p)
        ids = torch.cat([ids, next_id], dim=-1)
        times.append(time.process_time() - start)
    return times


_ = time_naive_per_token("warmup", 4, 0.7, 50, 1.0)  # discard the cold-start cost
naive_times = time_naive_per_token("The capital of France is", 200, 0.7, 50, 1.0)
print("Naive total time for 200 tokens:", sum(naive_times))



# ---------------------------------------------------------------
# Exercise 19: rewrite with past_key_values, feed only the newest token
# ---------------------------------------------------------------
def generate_cached(prompt, num_new_tokens, temperature, top_k, top_p):
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    generated = input_ids

    with torch.inference_mode():
        # Prefill: the ONLY time the whole prompt goes through the model
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_id = sample(outputs.logits, temperature, top_k, top_p)
        generated = torch.cat([generated, next_token_id], dim=-1)  # don't skip this

        # Decode: feed ONLY the newest token each step, reuse the cache
        for _ in range(num_new_tokens - 1):
            if next_token_id.item() == EOS_ID:
                break
            outputs = model(next_token_id, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_id = sample(outputs.logits, temperature, top_k, top_p)
            generated = torch.cat([generated, next_token_id], dim=-1)

    return tokenizer.decode(generated[0])


# Correctness check: at temperature=0, cached MUST equal naive exactly
torch.manual_seed(0)
naive_out = generate_naive("The capital of France is", 30, temperature=0, top_k=1, top_p=1.0)
torch.manual_seed(0)
cached_out = generate_cached("The capital of France is", 30, temperature=0, top_k=1, top_p=1.0)
assert naive_out == cached_out, "cache is producing a different sequence -- bug in cache handling"
print("Correctness check passed: naive and cached outputs match exactly.")


# ---------------------------------------------------------------
# Exercise 20: re-time WITH the cache, plot both curves together
# ---------------------------------------------------------------
def time_cached_per_token(prompt, num_new_tokens, temperature, top_k, top_p):
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    times = []
    with torch.inference_mode():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_id = sample(outputs.logits, temperature, top_k, top_p)
        for _ in range(num_new_tokens - 1):
            start = time.process_time()
            outputs = model(next_token_id, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_id = sample(outputs.logits, temperature, top_k, top_p)
            times.append(time.process_time() - start)
    return times, past_key_values


_, _ = time_cached_per_token("warmup", 4, 0.7, 50, 1.0)  # warm up this path too
cached_times, past_key_values = time_cached_per_token("The capital of France is", 200, 0.7, 50, 1.0)
print("Cached total time for ~200 tokens:", sum(cached_times))

speedup_at_200 = naive_times[199] / cached_times[-1]
print(f"Speedup at 200 tokens: {speedup_at_200:.2f}x")

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(naive_times) + 1), naive_times, label="naive", linewidth=1)
plt.plot(range(1, len(cached_times) + 1), cached_times, label="cached", linewidth=1)
plt.xlabel("token index")
plt.ylabel("time for this token (s)")
plt.title("Naive vs. cached generation: time per token")
plt.legend()
plt.savefig("../timing.png")
plt.show()


# ---------------------------------------------------------------
# Exercise 21: inspect the cache object's structure
# ---------------------------------------------------------------
print("Cache type:", type(past_key_values))
# Hint: recent `transformers` returns a DynamicCache object, not a plain
# tuple -- if past_key_values[0][0] errors for you, that is why, not a bug.
try:
    n_layers_seen = len(past_key_values)
    layer0_key_shape = past_key_values[0][0].shape  # older tuple-of-tuples style
except Exception as e:
    print("tuple-style indexing failed, trying DynamicCache style:", e)
    n_layers_seen = len(past_key_values.key_cache)
    layer0_key_shape = past_key_values.key_cache[0].shape

print("Layers in cache:", n_layers_seen, "-- should equal model.config.n_layer =", model.config.n_layer)
print("One layer's key tensor shape:", layer0_key_shape)



# ---------------------------------------------------------------
# Exercise 22: cache size in bytes for a 500-token sequence
# ---------------------------------------------------------------
n_layer = model.config.n_layer      # 12
n_head = model.config.n_head        # 12
head_dim = model.config.n_embd // model.config.n_head  # 64
seq_len_500 = 500
bytes_per_element = 4  # GPT-2 is float32 -- NOT the 2 bytes most formulas assume

cache_bytes = 2 * n_layer * n_head * head_dim * seq_len_500 * bytes_per_element  # 2 = key AND value
print(f"Computed cache size for {seq_len_500} tokens: {cache_bytes:,} bytes ({cache_bytes / 1e6:.2f} MB)")


# ---------------------------------------------------------------
# Exercise 23: extrapolate to an 8B model, 8192 tokens, 32 requests
# ---------------------------------------------------------------
# Fill in real numbers from an actual 8B-class model's config.json --
# don't guess n_layer/n_head/n_embd, they vary a lot between models, and
# check num_key_value_heads specifically since most modern 8B models use
# grouped-query attention (kv_heads < n_head), unlike GPT-2.
n_layer_8b = None
n_kv_heads_8b = None     # use num_key_value_heads if the model has GQA
head_dim_8b = None
bytes_per_element_8b = 2  # most modern 8B models run float16/bfloat16

if all(v is not None for v in [n_layer_8b, n_kv_heads_8b, head_dim_8b]):
    per_request_bytes = 2 * n_layer_8b * n_kv_heads_8b * head_dim_8b * 8192 * bytes_per_element_8b
    total_bytes = per_request_bytes * 32
    print(f"Cache per request: {per_request_bytes / 1e9:.2f} GB")
    print(f"Cache for 32 concurrent requests: {total_bytes / 1e9:.2f} GB")
else:
    print("Fill in n_layer_8b / n_kv_heads_8b / head_dim_8b from a real model's config first.")