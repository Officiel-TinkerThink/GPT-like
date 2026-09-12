/*
 * Minimal multi-series SVG line chart with a shared-x crosshair tooltip.
 *
 *   const chart = LineChart.create(svgEl, tooltipEl, legendEl, {
 *     yMin: 0, yMax: 1, yFormat: v => Math.round(v*100)+'%', xFormat: v => v.toLocaleString(),
 *   });
 *   chart.draw([{ name, color, points: [[x, y], …] }, …]);
 */
(function (root) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";

  function create(svg, tipEl, legendEl, opts) {
    const o = Object.assign({ yMin: 0, yMax: 1, yFormat: (v) => v, xFormat: (v) => v, xLabel: "", refLines: [] }, opts);
    let series = [];

    function el(tag, attrs, parent) {
      const n = document.createElementNS(NS, tag);
      for (const k in attrs) n.setAttribute(k, attrs[k]);
      (parent || svg).appendChild(n);
      return n;
    }

    function draw(s) {
      series = s;
      const W = svg.clientWidth || 800, H = svg.clientHeight || 280;
      const m = { l: 48, r: 60, t: 14, b: 30 };
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      svg.innerHTML = "";
      const xs = series.flatMap((q) => q.points.map((p) => p[0]));
      const xMax = Math.max(1, ...xs);
      const x = (v) => m.l + (v / xMax) * (W - m.l - m.r);
      const y = (v) => m.t + (1 - (v - o.yMin) / (o.yMax - o.yMin)) * (H - m.t - m.b);

      for (let t = 0; t <= 4; t++) {
        const v = o.yMin + ((o.yMax - o.yMin) * t) / 4;
        el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: "grid-line" });
        const txt = el("text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end", class: "axis-text" });
        txt.textContent = o.yFormat(v);
      }
      [0, 0.25, 0.5, 0.75, 1].forEach((f) => {
        const txt = el("text", { x: x(xMax * f), y: H - 8, "text-anchor": f === 0 ? "start" : f === 1 ? "end" : "middle", class: "axis-text" });
        txt.textContent = o.xFormat(Math.round(xMax * f)) + (f === 1 && o.xLabel ? " " + o.xLabel : "");
      });

      for (const r of o.refLines) {
        el("line", { x1: m.l, x2: W - m.r, y1: y(r.y), y2: y(r.y), class: "ref-line" });
        const t = el("text", { x: W - m.r - 4, y: y(r.y) - 5, "text-anchor": "end", class: "axis-text" }); t.textContent = r.label;
      }
      const ends = [];
      series.forEach((q) => {
        if (!q.points.length) return;
        const d = q.points.map((p, i) => (i ? "L" : "M") + x(p[0]) + "," + y(p[1])).join("");
        el("path", { d, class: "series", stroke: q.color });
        const last = q.points[q.points.length - 1];
        ends.push({ q, x: x(last[0]), y: y(last[1]), label: o.yFormat(last[1]) });
      });
      // end-of-line value labels, nudged apart so they never overlap
      ends.sort((a, b) => a.y - b.y);
      for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 14) ends[i].y = ends[i - 1].y + 14;
      ends.forEach((e) => {
        el("circle", { cx: e.x, cy: e.y, r: 3.5, fill: e.q.color });
        const lbl = el("text", { x: e.x + 8, y: e.y + 4, class: "end-label" });
        lbl.textContent = e.label;
      });

      // legend
      if (legendEl) {
        legendEl.innerHTML = series.map((q) => `<span class="legend-item"><i style="background:${q.color}"></i>${q.name}</span>`).join("");
      }

      // hover
      const cross = el("line", { x1: 0, x2: 0, y1: m.t, y2: H - m.b, class: "crosshair", visibility: "hidden" });
      const markers = series.map((q) => el("circle", { r: 4.5, class: "marker", fill: q.color, visibility: "hidden" }));
      svg.onmousemove = (e) => {
        if (!series.length || !series[0].points.length) return;
        const rect = svg.getBoundingClientRect();
        const px = ((e.clientX - rect.left) / rect.width) * W;
        const xv = ((px - m.l) / (W - m.l - m.r)) * xMax;
        // nearest point on the first series (all share the same checkpoints)
        const pts = series[0].points;
        let idx = 0, best = Infinity;
        pts.forEach((p, i) => { const d = Math.abs(p[0] - xv); if (d < best) { best = d; idx = i; } });
        const gx = pts[idx][0];
        cross.setAttribute("x1", x(gx)); cross.setAttribute("x2", x(gx)); cross.setAttribute("visibility", "visible");
        let rows = "";
        series.forEach((q, i) => {
          const p = q.points[idx];
          if (!p) { markers[i].setAttribute("visibility", "hidden"); return; }
          markers[i].setAttribute("cx", x(p[0])); markers[i].setAttribute("cy", y(p[1])); markers[i].setAttribute("visibility", "visible");
          rows += `<div><i style="background:${q.color}"></i>${q.name} <b>${o.yFormat(p[1])}</b></div>`;
        });
        tipEl.hidden = false;
        tipEl.innerHTML = `<div class="tip-title">${o.xFormat(gx)} ${o.xLabel}</div>${rows}`;
        const wr = svg.parentElement.getBoundingClientRect();
        const left = e.clientX - wr.left;
        tipEl.style.left = Math.min(left, wr.width - tipEl.offsetWidth - 8) + "px";
        tipEl.style.top = (rect.top - wr.top + 10) + "px";
      };
      svg.onmouseleave = () => { cross.setAttribute("visibility", "hidden"); markers.forEach((mk) => mk.setAttribute("visibility", "hidden")); tipEl.hidden = true; };
    }

    let timer = null;
    window.addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(() => series.length && draw(series), 150); });

    return { draw };
  }

  root.LineChart = { create };
})(window);
