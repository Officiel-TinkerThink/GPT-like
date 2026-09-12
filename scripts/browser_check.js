// Smoke test of the served app: node scripts/browser_check.js (needs playwright-core; serves at 127.0.0.1:8776)
const { chromium } = require('playwright-core');
(async () => {
  const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1360, height: 900 } });
  const errs = []; p.on('pageerror', e => errs.push('PAGEERROR ' + e.message)); p.on('console', m => { if (m.type()==='error') errs.push('CONSOLE ' + m.text()); });
  await p.goto('http://127.0.0.1:8776/index.html');
  await p.waitForFunction(() => window.__gpt && window.__gpt.model && !window.__gpt.G.running && window.__gpt.G.gen.length > 0, null, { timeout: 60000 });
  await p.waitForTimeout(500);
  console.log('ckpts', await p.$$eval('#ckpt option', o => o.map(x => x.value)));
  console.log('stats', await p.textContent('#genStats'));
  console.log('memo', await p.textContent('#memo'));
  console.log('output', (await p.textContent('#output')).replace(/\s+/g, ' ').slice(0, 200));
  console.log('cands', (await p.textContent('#cands')).replace(/\s+/g, ' ').slice(0, 120));
  await p.screenshot({ path: 'gpt1.png' });
  // greedy
  await p.fill('#prompt', 'I had always thought Jack'); await p.evaluate(() => { const t = document.getElementById('temp'); t.value = 0; t.dispatchEvent(new Event('input')); });
  await p.click('#btnGen'); await p.waitForFunction(() => !window.__gpt.G.running && window.__gpt.G.gen.length > 5, null, { timeout: 60000 });
  console.log('greedy', (await p.textContent('#output')).replace(/\s+/g, ' ').slice(0, 200));
  // manual mode
  await p.check('#chkManual'); await p.click('#btnGen'); await p.waitForTimeout(800);
  console.log('manual cands', await p.$$eval('#cands .cand.pick', c => c.length));
  const first = await p.$('#cands .cand'); if (first) { await first.click(); await p.waitForTimeout(500); console.log('after pick gen len', await p.evaluate(() => window.__gpt.G.gen.length)); }
  await p.click('#btnStep'); await p.waitForTimeout(500); console.log('after +1 len', await p.evaluate(() => window.__gpt.G.gen.length));
  await p.uncheck('#chkManual');
  // epoch switch
  await p.selectOption('#ckpt', '1'); await p.waitForFunction(() => !window.__gpt.G.running && window.__gpt.G.gen.length > 5, null, { timeout: 60000 });
  console.log('ep1', (await p.textContent('#output')).replace(/\s+/g, ' ').slice(0, 160));
  await p.click('.tab[data-tab="attention"]'); await p.waitForTimeout(600);
  await p.hover('#diagram', { position: { x: 200, y: 200 } }); await p.waitForTimeout(300);
  console.log('thumbs', await p.$$eval('#thumbs button', x => x.length), 'tip', await p.textContent('#cellTip'));
  await p.screenshot({ path: 'gpt2.png', fullPage: true });
  await p.click('.tab[data-tab="training"]'); await p.waitForTimeout(500);
  console.log('train', (await p.textContent('#trainStats')).replace(/\s+/g, ' ').slice(0, 200));
  await p.hover('#chart', { position: { x: 300, y: 120 } }); await p.waitForTimeout(300);
  await p.screenshot({ path: 'gpt3.png', fullPage: true });
  await p.click('.tab[data-tab="tokenizer"]'); await p.waitForTimeout(300);
  await p.fill('#tokText', 'Hello, Jack Gisburn -- a cheap zebra!'); await p.waitForTimeout(200);
  console.log('tok', await p.textContent('#tokNote'));
  await p.screenshot({ path: 'gpt4.png', fullPage: true });
  await p.setViewportSize({ width: 400, height: 800 }); await p.click('.tab[data-tab="generate"]'); await p.waitForTimeout(300); await p.screenshot({ path: 'gpt5.png', fullPage: true });
  console.log('scrollW', await p.evaluate(() => document.documentElement.scrollWidth));
  console.log('errors', errs);
  await b.close();
})();
