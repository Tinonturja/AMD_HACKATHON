"""
Day 1, Part 2 -- What the model actually outputs.
Exercises 6-9.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()


def top10(prompt):
    ids = torch.tensor([tokenizer.encode(prompt)])
    with torch.no_grad():
        out = model(ids)
    last = out.logits[0, -1]
    probs = torch.softmax(last, dim=-1)
    top = torch.topk(probs, 10)
    return list(zip(top.values.tolist(), top.indices.tolist()))


# ---------------------------------------------------------------
# Exercise 6: top-10 for three prompts of increasing specificity
# ---------------------------------------------------------------
prompts = ["The capital of", "The capital of France is"]
for p in prompts:
    print(f"\nPrompt: {p!r}")
    for prob, idx in top10(p):
        print(f"  {prob:.4f}  {tokenizer.decode([idx])!r}")




# ---------------------------------------------------------------
# Exercise 7: how much probability mass do the top 10 tokens hold?
# ---------------------------------------------------------------
for p in prompts:
    mass = sum(prob for prob, _ in top10(p))
    print(f"{p!r:<30} top-10 mass = {mass:.4f}")



# ---------------------------------------------------------------
# Exercise 8: per-position confidence across a sentence
# ---------------------------------------------------------------
sentence = "Humpty Dumpty sat on a wall. Humpty Dumpty had a great fall"
ids = torch.tensor([tokenizer.encode(sentence)])
with torch.no_grad():
    out = model(ids)
probs = torch.softmax(out.logits[0], dim=-1)  # [seq_len, vocab]

for i in range(len(ids[0]) - 1):
    actual_next = ids[0, i + 1].item()
    confidence = probs[i, actual_next].item()
    token_str = tokenizer.decode([ids[0, i].item()])
    print(f"pos {i:2d} {token_str!r:<12} confidence in the ACTUAL next token: {confidence:.4f}")


# ---------------------------------------------------------------
# Exercise 9: perplexity on ordinary English vs technical dyeing text
# ---------------------------------------------------------------
def compute_perplexity(text, model, tok):
    ids = torch.tensor([tok.encode(text)])
    with torch.no_grad():
        out = model(ids, labels=ids)
    return torch.exp(out.loss).item()

english_text = "The weather today is sunny with a light breeze from the west."
dye_text = "Among these, adsorption is still practiced widely because the operation is simple."

print("Perplexity, English:", compute_perplexity(english_text, model, tokenizer))
print("Perplexity, dyeing text:", compute_perplexity(dye_text, model, tokenizer))
