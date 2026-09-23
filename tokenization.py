import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

"""
Day 1, Part 1 -- Tokens are not words.
Exercises 1-5.
"""

# ---------------------------------------------------------------
# Exercise 1: chars-per-token ratio for ten strings across domains
# ---------------------------------------------------------------
samples = {
    "English prose": "The quick brown fox jumps over the lazy dog while the sun sets slowly over the hills.",
    "Python code": "def add(a, b):\n    return a + b\n\nresult = add(2, 3)\nprint(result)",
    "Long number": "3.14159265358979",
    "Chemical name": "polyvinyl alcohol",
    "URL": "https://www.example.com/path/to/page?query=value&other=123",
    "Bengali (same sentence)": "দ্রুত বাদামী শিয়াল অলস কুকুরের উপর দিয়ে লাফ দেয়।",
    
}

def chars_per_token(text, tok):
    ids = tok.encode(text)
    return len(text) / len(ids), len(ids)

print(f"{'Domain':<25}{'Chars':<8}{'Tokens':<8}{'Chars/token':<12}")
for name, text in samples.items():
    ratio, n_tokens = chars_per_token(text, tokenizer)
    print(f"{name:<25}{len(text):<8}{n_tokens:<8}{ratio:<12.2f}")

# Hint: the LOWEST ratio is the "worst" tokenization -- fewer characters
# per token means the tokenizer needed MORE tokens to say the same thing.
# Bengali will almost certainly be worst: GPT-2's vocabulary was built
# overwhelmingly from English text, so non-Latin scripts get chopped into
# many small byte-level pieces.


# ---------------------------------------------------------------
# Exercise 2: case sensitivity -- "cat" vs " cat" vs "Cat" vs "CAT" vs "cats"
# ---------------------------------------------------------------
words = ["cat", " cat", "Cat", "CAT", "cats"]
for w in words:
    ids = tokenizer.encode(w)
    print(f"{w!r:<8} -> {ids} ({len(ids)} token{'s' if len(ids) != 1 else ''})")

# Hint: count how many come back as exactly ONE token. Whichever ones
# don't tells you the model treats that exact surface form as less
# "familiar" -- this is the leading-space + case-sensitivity effect from
# the Concepts section: "cat" and " cat" are different ids, and
# capitalization variants often split into more pieces because they're
# rarer in training data.


# ---------------------------------------------------------------
# Exercise 3: find a short English word GPT-2 splits into 3+ tokens
# ---------------------------------------------------------------
candidates = ["certificate", "unbelievably", "photosynthesis", "onomatopoeia"]
for w in candidates:
    ids = tokenizer.encode(w)
    pieces = [tokenizer.decode([i]) for i in ids]
    print(f"{w:<16} -> {len(ids)} tokens: {pieces}")

# Hint: once you find one with 3+ pieces, write WHY in a comment: rare
# words aren't a single unit in GPT-2's byte-pair-encoding vocabulary, so
# BPE falls back to smaller, more common sub-pieces it does have.


# ---------------------------------------------------------------
# Exercise 4: does a string fit in GPT-2's 1024-token context window?
# ---------------------------------------------------------------
def check_string_inlimit(statement: str, tok, limit: int = 1024):
    n = len(tok.encode(statement))
    return {"tokens": n, "fits": n <= limit, "room_left": limit - n}

print(check_string_inlimit("The capital of France is Paris.", tokenizer))


# ---------------------------------------------------------------
# Exercise 5: chars-per-token ratio on your own dyeing-domain text
# ---------------------------------------------------------------
dye_para = (
    "Among these, adsorption is still practiced widely because the "
    "operation is simple and the reagents are inexpensive compared to "
    "advanced oxidation processes."
)  # replace this with an actual paragraph from one of your own papers
ratio, n_tokens = chars_per_token(dye_para, tokenizer)
print(f"Dyeing text: {len(dye_para)} chars, {n_tokens} tokens, ratio {ratio:.2f}")
# Hint: compare this to the "English prose" ratio from exercise 1.
# Technical/domain vocabulary usually tokenises LESS efficiently than
# plain English -- record this number, the worksheet says you'll need it
# again for the Week 2 RAG work.