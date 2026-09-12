"""Export tiny-GPT checkpoints for the browser app.

    python tools/export_web.py checkpoints docs/model

For each checkpoints/tiny_ep*.pt writes docs/model/ep<N>.bin (int8 per-row matrices, fp32 vectors) and a shared
docs/model/manifest.json (config, vocab, tensor table, training history, reference outputs for tests).
"""
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from model.gpt_model import GPTModel  # noqa: E402
from tokenizer.tokenizer import SimpleTokenizerV2  # noqa: E402
from tools.infer import generate  # noqa: E402

PROMPT = "I had always thought Jack Gisburn"  # every word is in the story's vocabulary

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
files = sorted(glob.glob(f"{src}/tiny_ep*.pt"), key=lambda f: int(re.search(r"ep(\d+)", f).group(1)))
manifest = None; samples = {}
for f in files:
    ck = torch.load(f, map_location="cpu")
    cfg, vocab, sd = ck["config"], ck["vocab"], ck["state_dict"]
    blob = bytearray(); table = []

    def add(name, arr, quant):
        global blob
        arr = np.ascontiguousarray(arr.detach().numpy().astype(np.float32))
        e = {"name": name, "shape": list(arr.shape), "offset": len(blob)}
        if quant and arr.ndim == 2:
            scale = np.abs(arr).max(axis=1, keepdims=True) / 127.0; scale[scale == 0] = 1.0
            q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
            e.update({"dtype": "int8", "scale_offset": len(blob) + q.nbytes}); blob += q.tobytes() + scale.astype(np.float32).tobytes()
        else:
            e["dtype"] = "f32"; blob += arr.tobytes()
        while len(blob) % 4: blob += b"\0"
        table.append(e)

    add("tok_emb", sd["tok_emb.weight"], True); add("pos_emb", sd["pos_emb.weight"], False)
    for l in range(cfg["n_layers"]):
        p = f"trf_blocks.{l}."
        add(f"l{l}.n1.w", sd[p + "norm1.scale"], False); add(f"l{l}.n1.b", sd[p + "norm1.shift"], False)
        add(f"l{l}.q", sd[p + "att.W_query.weight"], True); add(f"l{l}.k", sd[p + "att.W_key.weight"], True); add(f"l{l}.v", sd[p + "att.W_value.weight"], True)
        add(f"l{l}.o.w", sd[p + "att.out_proj.weight"], True); add(f"l{l}.o.b", sd[p + "att.out_proj.bias"], False)
        add(f"l{l}.n2.w", sd[p + "norm2.scale"], False); add(f"l{l}.n2.b", sd[p + "norm2.shift"], False)
        add(f"l{l}.f1.w", sd[p + "ff.layers.0.weight"], True); add(f"l{l}.f1.b", sd[p + "ff.layers.0.bias"], False)
        add(f"l{l}.f2.w", sd[p + "ff.layers.2.weight"], True); add(f"l{l}.f2.b", sd[p + "ff.layers.2.bias"], False)
    add("fn.w", sd["final_norm.scale"], False); add("fn.b", sd["final_norm.shift"], False)
    add("out", sd["out_head.weight"], True)
    ep = ck["epoch"]
    open(f"{out}/ep{ep}.bin", "wb").write(blob)
    # milestone sample for the Training tab (same prompt/seed for every checkpoint)
    model = GPTModel(cfg); model.load_state_dict(sd); model.eval(); tok = SimpleTokenizerV2(vocab); torch.manual_seed(123)
    ids = tok.encode(PROMPT); gen = generate(model, torch.tensor([ids]), 40, cfg["context_length"], temperature=0.8, top_k=20)[0].tolist()
    samples[ep] = tok.decode(gen[len(ids):])
    if manifest is None:
        # reference outputs from the LAST checkpoint are added below; tensor table is identical across checkpoints
        manifest = {"config": cfg, "vocab": vocab, "tensors": table, "checkpoints": []}
    manifest["checkpoints"].append({"epoch": ep, "file": f"ep{ep}.bin", "bytes": len(blob)})
    print(f, "->", f"{out}/ep{ep}.bin", f"{len(blob)/1e6:.2f} MB")

# reference: greedy next-token logits for a prompt with the final checkpoint
ck = torch.load(files[-1], map_location="cpu"); model = GPTModel(ck["config"]); model.load_state_dict(ck["state_dict"]); model.eval()
tok = SimpleTokenizerV2(ck["vocab"]); prompt = PROMPT; ids = tok.encode(prompt)
with torch.no_grad():
    logits = model(torch.tensor([ids]))[0]
top = torch.topk(logits[-1], 5)
manifest["reference"] = {"epoch": ck["epoch"], "prompt": prompt, "ids": ids, "top5": [[int(i), float(v)] for v, i in zip(top.values, top.indices)], "logit_row_sum": float(logits[-1].sum())}
hist = json.load(open(f"{src}/history.json"))
for e in hist["epochs"]:
    e.pop("sample", None)
    if e["epoch"] in samples: e["sample"] = samples[e["epoch"]]
manifest["history"] = hist["epochs"]; manifest["params"] = hist["params"]; manifest["tokens"] = hist["tokens"]; manifest["sample_prompt"] = PROMPT
json.dump(manifest, open(f"{out}/manifest.json", "w"))
print("manifest written;", len(files), "checkpoints")
