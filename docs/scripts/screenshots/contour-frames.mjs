// Contour-plot frames of the model learning: sessions with the first n of the take's 22 experiments.
// THEME=light|dark (default dark), NS=10,22, ZRANGE=lo,hi, OUT=<folder>; needs the API on port 8127.
import { chromium } from "playwright";
import { readFileSync, mkdirSync } from "node:fs";
const ORIGIN = "http://127.0.0.1:8127", API = `${ORIGIN}/api/v1`;
const THEME = process.env.THEME ?? "dark";
const OUT = process.env.OUT ?? `/tmp/readme-options/frames-${THEME}`; mkdirSync(OUT, { recursive: true });
const take = JSON.parse(readFileSync(process.argv[2], "utf8"));
const rows = take.experiments.data;
const VARS = [["Temperature", 200, 300, "°C"], ["Pressure", 20, 80, "bar"], ["H2_CO2", 2, 4, "mol/mol"], ["WHSV", 1.5, 18, "h⁻¹"]];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function api(path, body, method = "POST") {
  const r = await fetch(`${API}${path}`, { method, headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
  if (!r.ok) throw new Error(`${method} ${path} ${r.status}: ${(await r.text()).slice(0, 200)}`);
  return r.json();
}
async function makeSession(n) {
  const { session_id } = await api("/sessions", { name: `Loop after ${n} experiments` });
  for (const [name, min, max, unit] of VARS) await api(`/sessions/${session_id}/variables`, { name, type: "real", min, max, unit });
  const exps = rows.slice(0, n).map((r) => ({ inputs: Object.fromEntries(VARS.map(([v]) => [v, r[v]])), output: r.Output }));
  await api(`/sessions/${session_id}/experiments/batch`, { experiments: exps });
  await api(`/sessions/${session_id}/model/train`, { backend: "botorch", kernel: "Matern", kernel_params: { nu: 2.5 }, input_transform: "normalize", output_transform: "standardize", calibration_enabled: false });
  return session_id;
}
const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1680, height: 1050 }, deviceScaleFactor: 2, colorScheme: THEME });
await context.addInitScript((t) => { try { localStorage.setItem("theme", t); } catch {} }, THEME);
const page = await context.newPage();
// ZRANGE=lo,hi pins the app's Fixed colour scale to one range for every frame (it reads output_range from the contour response)
if (process.env.ZRANGE) {
  const [lo, hi] = process.env.ZRANGE.split(",").map(Number);
  await page.route(/\/visualizations\/contour/, async (route) => {
    const res = await route.fetch(); const body = await res.json(); body.output_range = [lo, hi];
    await route.fulfill({ response: res, json: body });
  });
}
const center = () => page.locator("main > div, #root > div").locator("xpath=//*[contains(., 'Model Visualizations')]").last();
const ns = (process.env.NS ?? "10,11,12,13,14,15,16,17,18,19,20,21,22").split(",").map(Number);
for (const n of ns) {
  const id = await makeSession(n);
  await page.goto(`${ORIGIN}/?session=${id}`);
  await page.getByRole("button", { name: "Model", exact: true }).waitFor({ timeout: 20000 });
  const banner = page.getByText("Unsaved work detected");
  if (await banner.isVisible().catch(() => false)) await page.getByRole("button", { name: "Discard" }).click();
  await page.getByRole("button", { name: "Show Visualizations" }).click();
  await page.getByRole("button", { name: "Plot Contour", exact: true }).click();
  await page.locator(".js-plotly-plot").first().waitFor({ timeout: 60000 });
  await sleep(1200);
  // axes: X Temperature, Y Pressure (defaults are the first two variables; set anyway)
  const sels = page.locator("select");
  for (let i = 0; i < (await sels.count()); i++) {
    const opts = await sels.nth(i).locator("option").allTextContents();
    if (opts.join("|") === "Temperature|Pressure|H2_CO2|WHSV") {
      const label = await sels.nth(i).evaluate((el) => el.closest("div")?.innerText ?? "");
      if (/X-Axis/.test(label)) await sels.nth(i).selectOption("Temperature");
      if (/Y-Axis/.test(label)) await sels.nth(i).selectOption("Pressure");
    }
  }
  for (const [name, value] of [["H2_CO2", "4"], ["WHSV", "1.5"]]) {
    const box = page.getByLabel(`${name} (value)`, { exact: true });
    await box.fill(value); await box.press("Enter"); await sleep(400);
  }
  const pts = page.getByLabel("Show Experimental Points");
  if (!(await pts.isChecked())) await pts.click();
  const nxt = page.getByLabel("Show Next Point");
  if ((await nxt.count()) && (await nxt.isChecked())) await nxt.click();
  await sleep(2500);
  const gd = page.locator(".js-plotly-plot").first();
  await gd.evaluate((el) => el.scrollIntoView({ block: "center", behavior: "instant" }));
  await sleep(600);
  await gd.screenshot({ path: `${OUT}/contour-${String(n).padStart(2, "0")}.png` });
  const range = await gd.evaluate((el) => { const t = el._fullData?.find((d) => d.type === "contour"); return t ? [t.zmin, t.zmax] : null; });
  console.log(`n=${n} session ${id} zrange ${JSON.stringify(range)}`);
  await api(`/sessions/${id}`, null, "DELETE").catch(() => {});
}
await browser.close();
