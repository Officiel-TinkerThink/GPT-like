// node --test tests/  — checks the browser inference engine (docs/js/gpt.js) against the exported PyTorch model.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const GPT = require("../docs/js/gpt.js");

const MODEL = path.join(__dirname, "..", "docs", "model");
const manifest = JSON.parse(fs.readFileSync(path.join(MODEL, "manifest.json"), "utf8"));
const tok = new GPT.Tokenizer(manifest.vocab);
function loadModel(file) {
  const b = fs.readFileSync(path.join(MODEL, file));
  return new GPT.Model(manifest, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}
const last = manifest.checkpoints[manifest.checkpoints.length - 1];

test("tokenizer matches the Python SimpleTokenizerV2 on the reference prompt", () => {
  assert.deepEqual(tok.encode(manifest.reference.prompt), manifest.reference.ids);
  assert.equal(tok.size, Object.keys(manifest.vocab).length);
});

test("tokenizer: unknown words map to <|unk|> and decode re-attaches punctuation", () => {
  const ids = tok.encode('Hello, zzzqqq world -- "yes".');
  assert.ok(ids.includes(tok.unk));
  assert.equal(tok.decode(tok.encode("I said, no.")), "I said, no.");
  assert.deepEqual(tok.tokens("zzzqqq is"), ["<|unk|>", "is"]);
});

test("JS forward pass reproduces PyTorch top-5 logits (int8 tolerance)", () => {
  const m = loadModel(last.file);
  assert.equal(last.epoch, manifest.reference.epoch);
  const logits = m.prime(manifest.reference.ids);
  const ref = manifest.reference.top5;
  const jsTop = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, 5);
  assert.equal(jsTop[0], ref[0][0], "greedy next token differs");
  for (const [id, v] of ref) assert.ok(Math.abs(logits[id] - v) < 0.35, `logit ${id}: js ${logits[id]} vs torch ${v}`);
});

test("KV-cache step-by-step generation equals re-priming from scratch", () => {
  const m = loadModel(last.file);
  const ids = tok.encode(manifest.sample_prompt);
  const a = m.prime(ids); const next = GPT.Model.sample(a, 0).id; const b = m.push(next);
  const m2 = loadModel(last.file); const c = m2.prime(ids.concat([next]));
  for (let i = 0; i < b.length; i++) assert.ok(Math.abs(b[i] - c[i]) < 1e-3);
  assert.equal(m.ids.length, ids.length + 1);
});

test("attention rows are causal distributions summing to 1", () => {
  const m = loadModel(manifest.checkpoints[0].file);
  m.prime(tok.encode("I had always thought Jack Gisburn rather a cheap genius"));
  for (let l = 0; l < m.L; l++) for (let h = 0; h < m.NH; h++) {
    const rows = m.attn[l][h];
    rows.forEach((row, p) => { assert.equal(row.length, p + 1); const s = row.reduce((x, y) => x + y, 0); assert.ok(Math.abs(s - 1) < 1e-4); });
  }
});

test("context window: pushing past context_length re-primes on the last CTX tokens", () => {
  const m = loadModel(manifest.checkpoints[0].file);
  const ids = Array.from({ length: m.CTX }, (_, i) => i % 50);
  m.prime(ids); m.push(7);
  assert.equal(m.ids.length, m.CTX); assert.equal(m.ids[m.CTX - 1], 7); assert.equal(m.cache[0].K.length, m.CTX);
});

test("sampling: temperature 0 is greedy, top-k restricts candidates, probabilities sum to 1", () => {
  const logits = Float32Array.from([1, 5, 3, 4, 2]);
  assert.equal(GPT.Model.sample(logits, 0).id, 1);
  const r = GPT.Model.sample(logits, 1, 3, () => 0.999);
  assert.equal(r.candidates.length, 3); assert.deepEqual(r.candidates.map((c) => c[0]), [1, 3, 2]);
  assert.ok(Math.abs(r.candidates.reduce((s, c) => s + c[1], 0) - 1) < 1e-9);
  assert.equal(GPT.Model.sample(logits, 1, 3, () => 0).id, 1);
  const p = GPT.Model.softmax(logits, 0.5); assert.ok(p[1] > 0.7);
});
