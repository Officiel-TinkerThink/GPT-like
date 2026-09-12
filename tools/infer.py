"""Generate text from a trained checkpoint.

    python tools/infer.py --checkpoint checkpoints/tiny.pt --prompt "He was silent" --temperature 0.8 --top_k 20
"""
import argparse
import sys

import torch

sys.path.insert(0, ".")
from model.gpt_model import GPTModel  # noqa: E402
from tokenizer.tokenizer import SimpleTokenizerV2  # noqa: E402


def generate(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None):
    """Like utils.tokenize.generate_text_simple, but with temperature and top-k sampling."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :]
        if top_k:
            top = torch.topk(logits, top_k).values
            logits = torch.where(logits < top[:, [-1]], torch.full_like(logits, float("-inf")), logits)
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default="He was silent")
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=20)
    a = p.parse_args(argv)
    ck = torch.load(a.checkpoint, map_location="cpu")
    model = GPTModel(ck["config"]); model.load_state_dict(ck["state_dict"]); model.eval()
    tok = SimpleTokenizerV2(ck["vocab"])
    ids = torch.tensor([tok.encode(a.prompt)])
    out = generate(model, ids, a.max_new_tokens, ck["config"]["context_length"], a.temperature, a.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main(sys.argv[1:])
