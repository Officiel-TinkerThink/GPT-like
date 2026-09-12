"""Train a small GPT on the-verdict.txt with the repo's model and word-level tokenizer, saving checkpoints
that the browser app can load.

    python tools/train_tiny.py --epochs 300 --out checkpoints

Writes checkpoints/tiny_epN.pt for a few milestone epochs and checkpoints/history.json (loss curve + samples).
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, ".")
from model.gpt_model import GPTModel  # noqa: E402
from tokenizer.tokenizer import SimpleTokenizerV2  # noqa: E402
from utils.text_processing import read_file, build_vocab  # noqa: E402
from tools.infer import generate  # noqa: E402

CONFIG = {"vocab_size": None, "context_length": 64, "emb_dim": 192, "n_heads": 4, "n_layers": 4, "drop_rate": [0.1, 0.1, 0.1], "qkv_bias": False}
MILESTONES = [1, 3, 10, 30, 100, 300]


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--seed", type=int, default=123)
    a = p.parse_args(argv)
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    text = read_file("the-verdict.txt")
    vocab = build_vocab(text, ["<|endoftext|>", "<|unk|>"])
    tok = SimpleTokenizerV2(vocab)
    ids = tok.encode(text)
    split = int(0.9 * len(ids))
    train_ids, val_ids = ids[:split], ids[split:]
    cfg = dict(CONFIG, vocab_size=len(vocab))
    L = cfg["context_length"]

    def batches(seq, bs=8, shuffle=True):
        starts = list(range(0, len(seq) - L - 1, L // 2))
        if shuffle:
            starts = [starts[i] for i in torch.randperm(len(starts)).tolist()]
        for i in range(0, len(starts), bs):
            chunk = starts[i:i + bs]
            x = torch.tensor([seq[s:s + L] for s in chunk]); y = torch.tensor([seq[s + 1:s + L + 1] for s in chunk])
            yield x, y

    model = GPTModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"vocab {len(vocab)} · tokens {len(ids)} · params {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)
    history = {"config": cfg, "params": n_params, "tokens": len(ids), "vocab_size": len(vocab), "epochs": [], "milestones": MILESTONES}
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train(); tl = 0; n = 0
        for x, y in batches(train_ids):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten())
            loss.backward(); opt.step(); tl += loss.item(); n += 1
        model.eval(); vl = 0; m = 0
        with torch.no_grad():
            for x, y in batches(val_ids, shuffle=False):
                vl += torch.nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten()).item(); m += 1
        rec = {"epoch": epoch, "train_loss": tl / max(1, n), "val_loss": vl / max(1, m)}
        if epoch in MILESTONES or epoch == a.epochs:
            torch.manual_seed(a.seed)
            sample = tok.decode(generate(model, torch.tensor([tok.encode("I had always thought Jack Gisburn")]), 40, L, temperature=0.8, top_k=20)[0].tolist())
            rec["sample"] = sample
            torch.save({"config": cfg, "vocab": vocab, "state_dict": model.state_dict(), "epoch": epoch}, f"{a.out}/tiny_ep{epoch}.pt")
            print(f"epoch {epoch:4d} train {rec['train_loss']:.3f} val {rec['val_loss']:.3f} ({time.time()-t0:.0f}s) | {sample[:90]}")
        history["epochs"].append(rec)
    json.dump(history, open(f"{a.out}/history.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1:])
