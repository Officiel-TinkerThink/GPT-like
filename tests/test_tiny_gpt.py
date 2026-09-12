"""python -m unittest discover tests — tokenizer, model shapes and the sampling generator."""
import sys
import unittest

import torch

sys.path.insert(0, ".")
from model.gpt_model import GPTModel  # noqa: E402
from tokenizer.tokenizer import SimpleTokenizerV2  # noqa: E402
from tools.infer import generate  # noqa: E402
from utils.text_processing import build_vocab  # noqa: E402

TEXT = 'I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no great surprise to me.'
CFG = {"vocab_size": None, "context_length": 16, "emb_dim": 32, "n_heads": 4, "n_layers": 2, "drop_rate": [0.0, 0.0, 0.0], "qkv_bias": False}


class TokenizerTest(unittest.TestCase):
    def test_round_trip_and_unknown(self):
        vocab = build_vocab(TEXT, ["<|endoftext|>", "<|unk|>"])
        tok = SimpleTokenizerV2(vocab)
        plain = "It was no great surprise to me, said Jack; a cheap genius."
        tok = SimpleTokenizerV2(build_vocab(plain, ["<|endoftext|>", "<|unk|>"]))
        self.assertEqual(tok.decode(tok.encode(plain)), plain)  # punctuation re-attached, "--" stays a spaced token
        tok = SimpleTokenizerV2(vocab)
        ids = tok.encode("Jack thought zzz.")
        self.assertIn(vocab["<|unk|>"], ids)
        self.assertEqual(tok.decode(ids), "Jack thought <|unk|>.")


class ModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.vocab = build_vocab(TEXT, ["<|endoftext|>", "<|unk|>"])
        cfg = dict(CFG, vocab_size=len(self.vocab))
        self.model = GPTModel(cfg).eval()
        self.cfg = cfg

    def test_forward_shape_uses_context_length(self):
        x = torch.randint(0, len(self.vocab), (2, self.cfg["context_length"]))
        self.assertEqual(tuple(self.model(x).shape), (2, self.cfg["context_length"], len(self.vocab)))

    def test_generate_greedy_is_deterministic_and_extends(self):
        idx = torch.tensor([[1, 2, 3]])
        a = generate(self.model, idx, 5, self.cfg["context_length"], temperature=0.0)
        b = generate(self.model, idx, 5, self.cfg["context_length"], temperature=0.0)
        self.assertEqual(tuple(a.shape), (1, 8)); self.assertTrue(torch.equal(a, b))

    def test_top_k_restricts_choices(self):
        torch.manual_seed(1)
        idx = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            logits = self.model(idx)[0, -1]
        allowed = set(torch.topk(logits, 2).indices.tolist())
        for _ in range(20):
            out = generate(self.model, idx, 1, self.cfg["context_length"], temperature=1.5, top_k=2)
            self.assertIn(int(out[0, -1]), allowed)

    def test_gelu_matches_torch(self):
        from utils.layer import GELU
        x = torch.linspace(-4, 4, 101)
        self.assertTrue(torch.allclose(GELU()(x), torch.nn.functional.gelu(x, approximate="tanh"), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
