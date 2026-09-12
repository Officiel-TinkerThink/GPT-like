// Usage: npm run serve (port 8080) or python3 -m http.server 8776 --directory docs, then: node scripts/record_demo.js (needs playwright-core + ffmpeg)
const { chromium } = require('playwright-core');
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
(async () => {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 }, recordVideo: { dir: 'assets/raw-video', size: { width: 1280, height: 800 } } });
  const page = await ctx.newPage();
  await page.addInitScript(() => { document.addEventListener('DOMContentLoaded', () => {
    const c = document.createElement('div'); c.style.cssText = 'position:fixed;z-index:9999;width:22px;height:22px;border-radius:50%;background:rgba(52,211,153,.45);border:2px solid #fff;pointer-events:none;transform:translate(-50%,-50%);transition:transform .08s;left:-100px;top:-100px;box-shadow:0 2px 8px rgba(0,0,0,.4)';
    document.body.appendChild(c); document.addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; });
    document.addEventListener('mousedown', () => { c.style.transform = 'translate(-50%,-50%) scale(.7)'; }); document.addEventListener('mouseup', () => { c.style.transform = 'translate(-50%,-50%) scale(1)'; }); }); });
  await page.goto('http://127.0.0.1:8776/index.html');
  await page.waitForFunction(() => window.__gpt && window.__gpt.model && !window.__gpt.G.running && window.__gpt.G.gen.length > 0, null, { timeout: 60000 });
  await sleep(1500);
  async function glideClick(sel, pause = 400) { const b = await page.locator(sel).first().boundingBox(); await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2, { steps: 14 }); await sleep(pause); await page.mouse.down(); await sleep(60); await page.mouse.up(); }
  await glideClick('#prompt', 300); await page.fill('#prompt', ''); await page.type('#prompt', 'I had always thought Jack Gisburn', { delay: 30 }); await page.keyboard.press('Enter');
  await page.waitForFunction(() => !window.__gpt.G.running && window.__gpt.G.gen.length > 5, null, { timeout: 60000 }); await sleep(1200);
  // manual steering
  await glideClick('#chkManual', 400); await sleep(300); await glideClick('#btnGen', 300); await sleep(900);
  for (const n of [2, 1, 3, 1]) { await glideClick(`#cands .cand:nth-child(${n})`, 350); await sleep(700); }
  await sleep(800);
  await glideClick('#chkManual', 300); await sleep(300);
  // earlier checkpoint
  await glideClick('#ckpt', 300); await page.selectOption('#ckpt', '3'); await page.waitForFunction(() => !window.__gpt.G.running && window.__gpt.G.gen.length > 5, null, { timeout: 60000 }); await sleep(1500);
  const eps = await page.$$eval('#ckpt option', o => o.map(x => Number(x.value))); await page.selectOption('#ckpt', String(Math.max(...eps))); await page.waitForFunction(() => !window.__gpt.G.running && window.__gpt.G.gen.length > 5, null, { timeout: 60000 }); await sleep(800);
  await glideClick('.tab[data-tab="attention"]', 400); await sleep(1000);
  const d = await page.locator('#diagram').boundingBox();
  await page.mouse.move(d.x + d.width * 0.2, d.y + d.height * 0.25, { steps: 6 }); await page.mouse.move(d.x + d.width * 0.7, d.y + d.height * 0.7, { steps: 40 }); await sleep(700);
  await glideClick('#thumbs button:nth-child(3)', 500); await sleep(1000);
  await page.selectOption('#layer', '3'); await sleep(1200);
  await glideClick('.tab[data-tab="training"]', 400); await sleep(800);
  const c = await page.locator('#chart').boundingBox();
  await page.mouse.move(c.x + c.width * 0.1, c.y + c.height * 0.5, { steps: 6 }); await page.mouse.move(c.x + c.width * 0.9, c.y + c.height * 0.5, { steps: 50 }); await sleep(600);
  await page.evaluate(() => document.getElementById('samples').scrollIntoView({ behavior: 'smooth', block: 'center' })); await sleep(1800);
  await ctx.close(); await browser.close(); console.log('recorded');
})();
