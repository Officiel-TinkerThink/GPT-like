/* GPT from Scratch — UI controller. Depends on gpt.js and chart.js */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  let manifest = null, tok = null, loadCheckpoint = null, models = {}, model = null, story = "", storyTokens = [];
  const G = { promptIds: [], gen: [], running: false, stop: false, cands: null, logits: null };

  // ------------------------------------------------------------ boot
  (async () => {
    try {
      const api = await window.GPT.load("model/"); manifest = api.manifest; tok = api.tokenizer; loadCheckpoint = api.loadCheckpoint;
      story = await (await fetch("data/the-verdict.txt")).text(); storyTokens = tok.tokens(story);
      const sel = $("ckpt"); sel.innerHTML = manifest.checkpoints.map((c) => `<option value="${c.epoch}">epoch ${c.epoch}</option>`).join(""); sel.value = manifest.checkpoints.some((c) => c.epoch === 30) ? 30 : manifest.checkpoints[manifest.checkpoints.length - 1].epoch;
      $("vocabSize").textContent = tok.size.toLocaleString() + " tokens";
      await useCheckpoint(Number(sel.value));
      renderTraining(); renderTokens();
      setupAttentionControls();
      toast("Model loaded. Press Generate — or tick “I pick each token” and steer it yourself.");
      generate();
    } catch (e) { console.error(e); toast("Could not load the model: " + e.message); }
  })();
  async function useCheckpoint(epoch) {
    if (!models[epoch]) { const c = manifest.checkpoints.find((x) => x.epoch === epoch); models[epoch] = await loadCheckpoint(c.file); }
    model = models[epoch];
  }

  // ------------------------------------------------------------ generation
  function currentText() { return tok.decode(G.promptIds.concat(G.gen.map((g) => g.id))); }
  function renderOutput(withCursor) {
    const promptTxt = tok.decode(G.promptIds);
    const parts = G.gen.map((g, i) => { const t = tok.intToStr.get(g.id); const p = g.p; const bg = p === null ? "transparent" : `rgba(16,185,129,${(0.08 + 0.45 * p).toFixed(2)})`; return `<span class="g ${i === G.gen.length - 1 ? "cur" : ""}" style="background:${bg}" title="${p === null ? "greedy" : "p = " + (p * 100).toFixed(1) + "%"}">${esc(t)}</span>`; });
    // join like the tokenizer: no space before punctuation
    let html = `<span class="p">${esc(promptTxt)}</span>`;
    G.gen.forEach((g, i) => { const t = tok.intToStr.get(g.id); html += (/^[,.:;?_"()']$/.test(t) ? "" : " ") + parts[i]; });
    $("output").innerHTML = html + (withCursor ? '<span class="cursor"></span>' : "");
    // memorisation meter: share of generated 4-grams that appear verbatim in the story
    const genTok = G.gen.map((g) => tok.intToStr.get(g.id));
    if (genTok.length >= 4) {
      const hay = storyTokens.join(""); let hit = 0, n = 0;
      for (let i = 0; i + 4 <= genTok.length; i++) { const g = genTok.slice(i, i + 4); if (g.filter((t) => /\w/.test(t)).length < 3) continue; n++; if (hay.includes(g.join(""))) hit++; }
      $("memo").textContent = n ? `Memorisation: ${Math.round((hit / n) * 100)}% of 4-word sequences appear verbatim in the story.` : "";
    } else $("memo").textContent = "";
  }
  function renderCands(cands, chosen) {
    if (!cands) { $("cands").innerHTML = ""; return; }
    const manual = $("chkManual").checked;
    $("cands").innerHTML = cands.slice(0, 15).map(([id, p]) => `<div class="cand ${manual ? "pick" : ""} ${id === chosen ? "chosen" : ""}" data-id="${id}"><span class="w">${esc(tok.intToStr.get(id))}</span><span class="track"><i style="width:${(p * 100).toFixed(1)}%"></i></span><span class="p">${(p * 100).toFixed(1)}%</span></div>`).join("");
    if (manual) $("cands").querySelectorAll(".cand").forEach((el) => el.addEventListener("click", () => pick(Number(el.dataset.id))));
  }
  function nextDistribution(logits) {
    const T = Number($("temp").value), k = Number($("topk").value);
    const s = window.GPT.Model.sample(logits, T || 1e-3, k);
    // for display use the temperature-softmax over the top-k (greedy shows the same list)
    return { cands: s.candidates, pick: T === 0 ? s.candidates[0][0] : s.id };
  }
  async function generate() {
    if (G.running || !model) return;
    const ids = tok.encode($("prompt").value.trim() || "Every");
    G.promptIds = ids; G.gen = []; G.running = true; G.stop = false;
    $("btnGen").disabled = true; $("btnStop").disabled = false;
    const t0 = performance.now();
    G.logits = model.prime(ids);
    renderOutput(true);
    const max = Number($("maxTok").value);
    if ($("chkManual").checked) { G.cands = nextDistribution(G.logits).cands; renderCands(G.cands, null); G.running = false; $("btnGen").disabled = false; $("btnStop").disabled = true; $("genStats").textContent = "Pick a token on the right."; return; }
    for (let i = 0; i < max && !G.stop; i++) { await stepOnce(); await sleep(35); }
    finishGen(t0);
  }
  function finishGen(t0) { G.running = false; $("btnGen").disabled = false; $("btnStop").disabled = true; renderOutput(false); $("genStats").textContent = `${G.gen.length} tokens · ${((performance.now() - (t0 || performance.now())) / 1000).toFixed(1)} s · epoch ${$("ckpt").value}`; if (document.querySelector('.tab-body[data-tab="attention"]').classList.contains("active")) renderDiagram(); }
  async function stepOnce(forcedId) {
    const d = nextDistribution(G.logits); const id = forcedId ?? d.pick; const pc = d.cands.find((c) => c[0] === id); const p = pc ? pc[1] : null;
    renderCands(d.cands, id);
    G.gen.push({ id, p }); G.logits = model.push(id); renderOutput(true);
    if (id === tok.eot) G.stop = true;
  }
  function pick(id) { if (!model || G.running) return; stepOnce(id).then(() => { G.cands = nextDistribution(G.logits).cands; renderCands(G.cands, null); renderOutput(true); $("genStats").textContent = `${G.gen.length} tokens · epoch ${$("ckpt").value}`; }); }
  async function plusOne() {
    if (!model || G.running) return;
    if (!G.promptIds.length) { G.promptIds = tok.encode($("prompt").value.trim() || "Every"); G.gen = []; G.logits = model.prime(G.promptIds); }
    await stepOnce(); G.cands = nextDistribution(G.logits).cands; renderCands(G.cands, null); finishGen();
  }

  // ------------------------------------------------------------ attention
  function setupAttentionControls() { $("layer").innerHTML = Array.from({ length: model.L }, (_, i) => `<option value="${i}">${i + 1}</option>`).join(""); $("head").innerHTML = Array.from({ length: model.NH }, (_, i) => `<option value="${i}">${i + 1}</option>`).join(""); }
  let geom = null;
  function drawMatrix(canvas, A, size, labels) {
    const n = A.length; const pad = labels ? Math.round(size * 0.17) : 0; const cell = (size - pad) / Math.max(1, n);
    canvas.width = size; canvas.height = size; const ctx = canvas.getContext("2d"); ctx.fillStyle = "#0b1512"; ctx.fillRect(0, 0, size, size);
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) { const v = Math.pow(A[i][j], 0.45); ctx.fillStyle = `rgb(${Math.round(20 + 215 * v)},${Math.round(30 + 225 * v)},${Math.round(25 + 190 * v)})`; ctx.fillRect(pad + j * cell, pad + i * cell, Math.ceil(cell), Math.ceil(cell)); }
    if (labels) { ctx.fillStyle = "#fff"; ctx.font = `${Math.max(8, Math.min(13, cell * 0.6))}px ui-monospace, Menlo, monospace`; ctx.textBaseline = "middle"; ctx.textAlign = "right"; labels.forEach((t, i) => ctx.fillText(t.slice(0, 12), pad - 6, pad + i * cell + cell / 2)); labels.forEach((t, j) => { ctx.save(); ctx.translate(pad + j * cell + cell / 2, pad - 6); ctx.rotate(-Math.PI / 2); ctx.textAlign = "left"; ctx.fillText(t.slice(0, 12), 0, 0); ctx.restore(); }); }
    return { pad, cell };
  }
  function contextTokens() { return model.ids.map((i) => tok.intToStr.get(i)); }
  function renderDiagram() {
    if (!model || !model.ids.length) return;
    const l = Number($("layer").value), h = Number($("head").value); const A = model.attn[l][h]; const labels = contextTokens();
    geom = drawMatrix($("diagram"), A, 600, labels);
    const th = $("thumbs"); th.innerHTML = "";
    model.attn[l].forEach((M, hi) => { const b = document.createElement("button"); b.className = hi === h ? "cur" : ""; const c = document.createElement("canvas"); drawMatrix(c, M, 96, null); b.appendChild(c); b.insertAdjacentHTML("beforeend", `<span>L${l + 1} H${hi + 1}</span>`); b.addEventListener("click", () => { $("head").value = hi; renderDiagram(); }); th.appendChild(b); });
  }
  $("diagram").addEventListener("mousemove", (e) => {
    if (!geom || !model || !model.ids.length) return; const r = $("diagram").getBoundingClientRect(); const sx = 600 / r.width; const px = (e.clientX - r.left) * sx, py = (e.clientY - r.top) * sx;
    const j = Math.floor((px - geom.pad) / geom.cell), i = Math.floor((py - geom.pad) / geom.cell); const n = model.ids.length; const tip = $("cellTip");
    if (i < 0 || j < 0 || i >= n || j > i) { tip.hidden = true; return; }
    const v = model.attn[Number($("layer").value)][Number($("head").value)][i][j]; const labels = contextTokens();
    tip.hidden = false; tip.textContent = `${labels[i]} → ${labels[j]}: ${(v * 100).toFixed(1)}%`; tip.style.left = (e.clientX - r.left + 12) + "px"; tip.style.top = (e.clientY - r.top + 12) + "px";
  });
  $("diagram").addEventListener("mouseleave", () => { $("cellTip").hidden = true; });

  // ------------------------------------------------------------ training tab
  const chart = window.LineChart.create($("chart"), $("tip"), $("legend"), { yMin: 0, yMax: 10, yFormat: (v) => v.toFixed(1), xFormat: (v) => String(v), xLabel: "epochs" });
  function renderTraining() {
    const H = manifest.history; const cfg = manifest.config;
    $("trainLead").textContent = `A ${(manifest.params / 1e6).toFixed(2)} M-parameter GPT (context ${cfg.context_length}, embedding ${cfg.emb_dim}, ${cfg.n_heads} heads × ${cfg.n_layers} layers) trained with AdamW on ${manifest.tokens.toLocaleString()} tokens — 90 % for training, 10 % held out. It starts by guessing uniformly over ${tok.size.toLocaleString()} words (loss ln ${tok.size} ≈ ${Math.log(tok.size).toFixed(1)}) and ends by reciting the training text.`;
    const last = H[H.length - 1]; const bestVal = H.reduce((a, b) => (b.val_loss < a.val_loss ? b : a));
    const tile = (l, v, sub) => `<div class="tile"><div class="tile-label">${l}</div><div class="tile-value">${v}</div>${sub ? `<div class="tile-sub">${sub}</div>` : ""}</div>`;
    $("trainStats").innerHTML = tile("Parameters", (manifest.params / 1e6).toFixed(2) + " M") + tile("Tokens", manifest.tokens.toLocaleString(), "one short story") + tile("Epochs", H.length) + tile("Final train loss", last.train_loss.toFixed(3), "perplexity " + Math.exp(last.train_loss).toFixed(1)) + tile("Best val loss", bestVal.val_loss.toFixed(3), "at epoch " + bestVal.epoch) + tile("Final val loss", last.val_loss.toFixed(3), "overfits after epoch " + bestVal.epoch);
    chart.draw([{ name: "Training loss", color: "#34d399", points: H.map((e) => [e.epoch, e.train_loss]) }, { name: "Validation loss", color: "#fbbf24", points: H.map((e) => [e.epoch, e.val_loss]) }]);
    $("samples").innerHTML = H.filter((e) => e.sample).map((e) => `<div class="sample"><b>epoch ${e.epoch}</b><span>${esc(e.sample)}</span></div>`).join("");
  }

  // ------------------------------------------------------------ tokenizer tab
  function renderTokens() {
    const words = tok.split($("tokText").value); let unk = 0;
    $("tokens").innerHTML = words.map((w) => { const known = w in tok.strToInt; if (!known) unk++; return `<span class="${known ? "" : "unk"}">${esc(w)}<small>${known ? tok.strToInt[w] : "unk"}</small></span>`; }).join("");
    $("tokNote").textContent = `${words.length} tokens, ${unk} unknown to the story's vocabulary.`;
  }

  // ------------------------------------------------------------ misc
  function toast(html, ms) { const t = document.createElement("div"); t.className = "toast"; t.innerHTML = html; $("toasts").appendChild(t); setTimeout(() => { t.classList.add("out"); setTimeout(() => t.remove(), 300); }, ms || 3200); }
  function switchTab(name) { document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name)); document.querySelectorAll(".tab-body").forEach((b) => b.classList.toggle("active", b.dataset.tab === name)); if (name === "attention") renderDiagram(); }
  document.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => b.blur()));
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));
  $("btnGen").addEventListener("click", generate);
  $("btnStop").addEventListener("click", () => { G.stop = true; });
  $("btnStep").addEventListener("click", plusOne);
  $("prompt").addEventListener("keydown", (e) => { if (e.key === "Enter") generate(); });
  $("temp").addEventListener("input", (e) => { $("tempOut").value = Number(e.target.value).toFixed(2); if (G.logits && !G.running) { G.cands = nextDistribution(G.logits).cands; renderCands(G.cands, null); } });
  $("topk").addEventListener("input", (e) => { $("topkOut").value = e.target.value; if (G.logits && !G.running) { G.cands = nextDistribution(G.logits).cands; renderCands(G.cands, null); } });
  $("chkManual").addEventListener("change", () => { if (G.logits && !G.running) renderCands(nextDistribution(G.logits).cands, null); });
  $("ckpt").addEventListener("change", async (e) => { $("btnGen").disabled = true; await useCheckpoint(Number(e.target.value)); $("btnGen").disabled = false; generate(); });
  $("layer").addEventListener("change", renderDiagram); $("head").addEventListener("change", renderDiagram);
  $("tokText").addEventListener("input", renderTokens);
  window.__gpt = { G, get model() { return model; }, generate, stepOnce, pick };
})();
