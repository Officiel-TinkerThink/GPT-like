/*
 * GPT inference in plain JavaScript — a port of model/gpt_model.py (pre-LayerNorm transformer blocks,
 * causal multi-head attention, GELU feed-forward, untied output head) with the repo's word-level tokenizer.
 * Keeps a KV cache for token-by-token generation and every head's attention weights for visualisation.
 * Browser global (window.GPT) + CommonJS module.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.GPT = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ------------------------------------------------------------ SimpleTokenizerV2
  class Tokenizer {
    constructor(vocab) { this.strToInt = vocab; this.intToStr = new Map(Object.entries(vocab).map(([s, i]) => [i, s])); this.unk = vocab["<|unk|>"]; this.eot = vocab["<|endoftext|>"]; }
    split(text) { return text.split(/([,.:;?_"()']|--|\s)/).map((t) => t.trim()).filter(Boolean); }
    encode(text) { return this.split(text).map((t) => (t in this.strToInt ? this.strToInt[t] : this.unk)); }
    tokens(text) { return this.split(text).map((t) => (t in this.strToInt ? t : "<|unk|>")); }
    decode(ids) { return ids.map((i) => this.intToStr.get(i) ?? "<|unk|>").join(" ").replace(/\s+([,.:;?_"()'])/g, "$1"); }
    get size() { return Object.keys(this.strToInt).length; }
  }

  // ------------------------------------------------------------ model
  class Model {
    constructor(manifest, buffer) {
      this.cfg = manifest.config; this.t = new Map();
      for (const e of manifest.tensors) { const n = e.shape.reduce((a, b) => a * b, 1); if (e.dtype === "int8") this.t.set(e.name, { shape: e.shape, q: new Int8Array(buffer, e.offset, n), s: new Float32Array(buffer, e.scale_offset, e.shape[0]) }); else this.t.set(e.name, { shape: e.shape, f: new Float32Array(buffer, e.offset, n) }); }
      this.D = this.cfg.emb_dim; this.NH = this.cfg.n_heads; this.HD = this.D / this.NH; this.L = this.cfg.n_layers; this.CTX = this.cfg.context_length;
      this.reset();
    }
    reset() { this.cache = Array.from({ length: this.L }, () => ({ K: [], V: [] })); this.attn = Array.from({ length: this.L }, () => Array.from({ length: this.NH }, () => [])); this.ids = []; }
    linear(name, x, bias) { const W = this.t.get(name); const [O, I] = W.shape; const y = new Float32Array(O); const b = bias ? this.t.get(bias).f : null; for (let o = 0; o < O; o++) { let a = 0; const off = o * I; for (let i = 0; i < I; i++) a += W.q[off + i] * x[i]; y[o] = a * W.s[o] + (b ? b[o] : 0); } return y; }
    ln(x, w, b) { const W = this.t.get(w).f, B = this.t.get(b).f, n = x.length; let m = 0; for (let i = 0; i < n; i++) m += x[i]; m /= n; let v = 0; for (let i = 0; i < n; i++) { const d = x[i] - m; v += d * d; } v /= n; const inv = 1 / Math.sqrt(v + 1e-5); const y = new Float32Array(n); for (let i = 0; i < n; i++) y[i] = (x[i] - m) * inv * W[i] + B[i]; return y; }
    static gelu(x) { const s = Math.sign(x), ax = Math.abs(x) / Math.SQRT2, t = 1 / (1 + 0.3275911 * ax); const erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax); return 0.5 * x * (1 + s * erf); }
    /** Process one token at position p (using the KV cache). Returns logits. */
    step(id, p) {
      const D = this.D, NH = this.NH, HD = this.HD; const E = this.t.get("tok_emb"), P = this.t.get("pos_emb").f;
      let x = new Float32Array(D); for (let i = 0; i < D; i++) x[i] = E.q[id * D + i] * E.s[id] + P[p * D + i];
      for (let l = 0; l < this.L; l++) {
        const h = this.ln(x, `l${l}.n1.w`, `l${l}.n1.b`);
        const q = this.linear(`l${l}.q`, h), k = this.linear(`l${l}.k`, h), v = this.linear(`l${l}.v`, h);
        const c = this.cache[l]; c.K.push(k); c.V.push(v); const n = c.K.length;
        const ctx = new Float32Array(D);
        for (let hh = 0; hh < NH; hh++) {
          const o = hh * HD; const sc = new Float32Array(n); let mx = -Infinity;
          for (let j = 0; j < n; j++) { let s = 0; const kj = c.K[j]; for (let d = 0; d < HD; d++) s += q[o + d] * kj[o + d]; s /= Math.sqrt(HD); sc[j] = s; if (s > mx) mx = s; }
          let sum = 0; for (let j = 0; j < n; j++) { sc[j] = Math.exp(sc[j] - mx); sum += sc[j]; } for (let j = 0; j < n; j++) sc[j] /= sum;
          this.attn[l][hh].push(Array.from(sc));
          for (let j = 0; j < n; j++) { const a = sc[j]; const vj = c.V[j]; for (let d = 0; d < HD; d++) ctx[o + d] += a * vj[o + d]; }
        }
        const ao = this.linear(`l${l}.o.w`, ctx, `l${l}.o.b`); for (let i = 0; i < D; i++) x[i] += ao[i];
        const h2 = this.ln(x, `l${l}.n2.w`, `l${l}.n2.b`); const f = this.linear(`l${l}.f1.w`, h2, `l${l}.f1.b`); for (let i = 0; i < f.length; i++) f[i] = Model.gelu(f[i]);
        const f2 = this.linear(`l${l}.f2.w`, f, `l${l}.f2.b`); for (let i = 0; i < D; i++) x[i] += f2[i];
      }
      return this.linear("out", this.ln(x, "fn.w", "fn.b"));
    }
    /** Feed a full prompt (rebuilding the cache) and return logits for the last position. */
    prime(ids) { this.reset(); const keep = ids.slice(-this.CTX); let logits = null; for (let p = 0; p < keep.length; p++) { logits = this.step(keep[p], p); this.ids.push(keep[p]); } return logits; }
    /** Append one token; if the context is full, re-prime on the last CTX tokens. */
    push(id) { if (this.ids.length >= this.CTX) return this.prime(this.ids.concat([id])); const logits = this.step(id, this.ids.length); this.ids.push(id); return logits; }
    static softmax(logits, temperature) { const T = Math.max(1e-6, temperature); let mx = -Infinity; for (const v of logits) if (v > mx) mx = v; const e = new Float64Array(logits.length); let s = 0; for (let i = 0; i < logits.length; i++) { e[i] = Math.exp((logits[i] - mx) / T); s += e[i]; } for (let i = 0; i < logits.length; i++) e[i] /= s; return e; }
    /** Sampling exactly like tools/infer.py: top-k filter on logits, then temperature softmax (temperature 0 = greedy). */
    static sample(logits, temperature, topK, rng) {
      let idx = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]);
      if (topK) idx = idx.slice(0, topK);
      if (!temperature) return { id: idx[0], probs: null };
      const sub = Float32Array.from(idx.map((i) => logits[i])); const p = Model.softmax(sub, temperature);
      let r = (rng || Math.random)(), acc = 0, pick = idx[idx.length - 1];
      for (let i = 0; i < idx.length; i++) { acc += p[i]; if (r < acc) { pick = idx[i]; break; } }
      return { id: pick, candidates: idx.map((id, i) => [id, p[i]]) };
    }
  }

  async function load(baseUrl, onProgress) {
    const manifest = await (await fetch(baseUrl + "manifest.json")).json();
    return { manifest, tokenizer: new Tokenizer(manifest.vocab), loadCheckpoint: async (file) => { const res = await fetch(baseUrl + file); const buf = await res.arrayBuffer(); if (onProgress) onProgress(buf.byteLength); return new Model(manifest, buf); } };
  }
  return { Tokenizer, Model, load };
});
