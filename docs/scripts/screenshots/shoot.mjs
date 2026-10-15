#!/usr/bin/env node
/**
 * Scripted screenshots of the ALchemist web app for the docs, on simulated data only.
 *
 * Starts the API (which serves the built web app) from this checkout with a fresh working directory,
 * drives the app in headless Chromium at 1680x1050, and writes every shot listed in shots.json to
 * docs/images/ twice: in the light theme (<name>.png) and, after switching the app to its dark theme and
 * back, in the dark theme (<name>-dark.png). The docs pages show the pair with .only-light / .only-dark.
 * On the way it asserts that every control name in controls.json is
 * visible at the step where the docs use it, and exits non-zero naming any that are missing.
 *
 * Data: the reactor simulator embedded in the docs (docs/tutorial/reactor/index.html, pinned commit in
 * its SOURCE.md) runs every experiment in a second tab, with a fixed noise seed. Setpoints come from the
 * app: an LHS design with a fixed seed, then BoTorch suggestions (deterministic for the same inputs).
 * Every simulator run is written to simulated-session.json next to this script. Never lab data.
 *
 * Usage (from this folder; `npm install` first):
 *   node shoot.mjs                  build the web bundle if stale, take every shot, check every control
 *   node shoot.mjs --check-pages    no browser: every **bold** name in the guide pages must be in controls.json,
 *                                   and every screenshot a page (or overrides/home.html) shows must be paired
 *                                   with its -dark.png (markup .only-light/.only-dark, and the file on disk)
 *   node shoot.mjs --no-build       fail instead of rebuilding a stale bundle
 *   node shoot.mjs --debug          also save a full-page screenshot after every step to $WORK/debug/
 *
 * Environment
 *   ALCHEMIST_PYTHON  Python that has ALchemist's dependencies (default ~/miniforge3/envs/alchemist-env/bin/python if it exists, else python3 on PATH)
 *   SIM_SEED          simulator noise seed (default 20261011)   LHS_SEED  initial design seed (default 42)
 *   WORK              parent of the run's scratch folder (default: the system temp folder); the API writes cache/ there
 *   PORT              API port (default: a free one; never 8000, which other local tools use)
 *   CHROME            browser executable (default: Playwright's own Chromium for the pinned version)
 */
import { execFileSync, spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { createServer, connect } from "node:net";
import { createServer as createHttpServer } from "node:http";
import { homedir, tmpdir } from "node:os";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "../../..");
const DOCS = join(REPO, "docs");
const WEB = join(REPO, "alchemist-web");
const IMAGES = join(DOCS, "images");
const SHOTS = JSON.parse(readFileSync(join(HERE, "shots.json"), "utf8")).shots;
const CONTROLS = JSON.parse(readFileSync(join(HERE, "controls.json"), "utf8")).controls;
const ARGS = new Set(process.argv.slice(2));
const DEBUG = ARGS.has("--debug");
const VIEWPORT = { width: 1680, height: 1050 };
const SIM_SEED = Number(process.env.SIM_SEED ?? 20261011);
const LHS_SEED = Number(process.env.LHS_SEED ?? 42);
const DOE_POINTS = 10;
const BATCH_Q = 4;
const LOGEI_ROUNDS = 2;
const DEFAULT_PYTHON = join(homedir(), "miniforge3/envs/alchemist-env/bin/python");
const PYTHON = process.env.ALCHEMIST_PYTHON ?? (existsSync(DEFAULT_PYTHON) ? DEFAULT_PYTHON : "python3");
// A release build (alchemist-web/.env.production) calls /api/v1 on the page's own origin, so it needs no
// rerouting. Only a dev-mode bundle (see .env.development) has this absolute origin baked in.
const BUILT_ORIGIN = "http://localhost:8000";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const t0 = Date.now();
const log = (msg) => process.stdout.write(`[${((Date.now() - t0) / 1000).toFixed(1).padStart(6)} s] ${msg}\n`);

// The reactor simulator's variables (alchemist-reactor-sim README; the first-loop tutorial uses these names)
const REACTOR_VARIABLES = [
  { name: "Temperature", min: 200, max: 300, unit: "°C", description: "Reactor temperature" },
  { name: "Pressure", min: 20, max: 80, unit: "bar", description: "Reactor pressure" },
  { name: "H2_CO2", min: 2, max: 4, unit: "mol/mol", description: "H2:CO2 feed ratio" },
  { name: "WHSV", min: 1.5, max: 18, unit: "h⁻¹", description: "Weight hourly space velocity" },
];

// ---------------------------------------------------------------------------------------------------
// --check-pages: bold names in the guide pages must all be in controls.json (no browser needed)
// ---------------------------------------------------------------------------------------------------
const PAGE_DIRS = ["guide", "get-started"];
function boldNamesInPages() {
  const found = [];
  const walk = (dir) => {
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      const p = join(dir, e.name);
      if (e.isDirectory()) walk(p);
      else if (e.name.endsWith(".md")) {
        let inFence = false;
        readFileSync(p, "utf8").split("\n").forEach((line, i) => {
          if (/^\s*(```|~~~)/.test(line)) { inFence = !inFence; return; }
          if (inFence) return;
          const noCode = line.replace(/`[^`]*`/g, "");
          for (const m of noCode.matchAll(/\*\*(.+?)\*\*/g)) found.push({ name: m[1], file: relative(DOCS, p), line: i + 1 });
        });
      }
    }
  };
  for (const d of PAGE_DIRS) if (existsSync(join(DOCS, d))) walk(join(DOCS, d));
  return found;
}
function checkPages() {
  const known = new Set(CONTROLS.flatMap((c) => [c.name, ...(c.aliases ?? [])]));
  const missing = boldNamesInPages().filter((b) => !known.has(b.name));
  if (missing.length) {
    console.error(`${missing.length} bold name(s) in the docs are not in controls.json:`);
    for (const m of missing) console.error(`  docs/${m.file}:${m.line}  **${m.name}**`);
    console.error("Add each to controls.json with the step where the app shows it (or un-bold it if it is not a control).");
    return false;
  }
  console.log(`All bold names in docs/${PAGE_DIRS.join(", docs/")} are in controls.json (${known.size} names).`);
  return true;
}

// Every app screenshot a page shows is shown as a light/dark pair: the light image with .only-light, the
// -dark.png sibling with .only-dark, and the -dark.png file exists (the hero README is exempt: PyPI cannot switch).
const SHOT_FILES = new Set(SHOTS.map((s) => s.file ?? join("guide", `${s.name}.png`)));
function pageSources() {
  const out = [];
  const walk = (dir) => {
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      const p = join(dir, e.name);
      if (e.isDirectory()) { if (!["_legacy", "scripts"].includes(e.name)) walk(p); }
      else if (e.name.endsWith(".md")) out.push(p);
    }
  };
  walk(DOCS);
  const home = join(REPO, "overrides/home.html");
  if (existsSync(home)) out.push(home);
  return out;
}
function checkDarkPairs() {
  const problems = [];
  const darkOf = (f) => f.replace(/\.png$/, "-dark.png");
  for (const p of pageSources()) {
    const rel = relative(REPO, p);
    // a reference is a Markdown image ![..](src){ attrs } or an <img ...> tag
    const refs = [];
    readFileSync(p, "utf8").split("\n").forEach((line, i) => {
      for (const m of line.matchAll(/!\[[^\]]*\]\(([^)\s]+)\)(\{[^}]*\})?/g)) refs.push({ src: m[1], attrs: m[2] ?? "", line: i + 1 });
      for (const m of line.matchAll(/<img\b[^>]*>/g)) {
        const src = m[0].match(/\bsrc="([^"]+)"/)?.[1];
        if (src) refs.push({ src, attrs: m[0], line: i + 1 });
      }
    });
    const shown = refs.filter((r) => /images\//.test(r.src)).map((r) => ({ ...r, file: r.src.slice(r.src.indexOf("images/") + "images/".length) }));
    for (const r of shown) {
      if (SHOT_FILES.has(r.file)) {
        if (!/only-light/.test(r.attrs)) problems.push(`${rel}:${r.line}  ${r.src} is an app screenshot but is not marked .only-light`);
        if (!shown.some((d) => d.file === darkOf(r.file) && /only-dark/.test(d.attrs))) problems.push(`${rel}:${r.line}  ${r.src} has no ${darkOf(r.file)} with .only-dark beside it`);
      } else if ([...SHOT_FILES].some((f) => darkOf(f) === r.file)) {
        if (!/only-dark/.test(r.attrs)) problems.push(`${rel}:${r.line}  ${r.src} is a dark screenshot but is not marked .only-dark`);
      }
    }
  }
  for (const f of SHOT_FILES) if (!existsSync(join(IMAGES, darkOf(f)))) problems.push(`docs/images/${darkOf(f)} is missing (run node shoot.mjs)`);
  if (problems.length) {
    console.error(`${problems.length} light/dark screenshot problem(s):`);
    for (const m of problems) console.error(`  ${m}`);
    console.error("Show each app screenshot as ![alt](x.png){ .only-light } followed by ![alt](x-dark.png){ .only-dark }, and take the dark file with shoot.mjs.");
    return false;
  }
  console.log(`Every app screenshot in the docs pages and overrides/home.html has its -dark.png pair (${SHOT_FILES.size} images).`);
  return true;
}
if (ARGS.has("--check-pages")) process.exit([checkPages(), checkDarkPairs()].every(Boolean) ? 0 : 1);

// ---------------------------------------------------------------------------------------------------
// Web bundle: rebuild when alchemist-web/dist is older than its sources
// ---------------------------------------------------------------------------------------------------
function newestMtime(p) {
  const s = statSync(p);
  if (!s.isDirectory()) return s.mtimeMs;
  let m = 0;
  for (const e of readdirSync(p)) m = Math.max(m, newestMtime(join(p, e)));
  return m;
}
function ensureBundle() {
  if (existsSync(join(REPO, "api/static/index.html"))) {
    throw new Error("api/static/ exists, so the API serves it instead of alchemist-web/dist. Remove it (it is a packaging copy) and rerun.");
  }
  const dist = join(WEB, "dist/index.html");
  const srcTime = Math.max(newestMtime(join(WEB, "src")), newestMtime(join(WEB, "index.html")), newestMtime(join(WEB, "package-lock.json")));
  if (existsSync(dist) && statSync(dist).mtimeMs >= srcTime) return;
  const why = existsSync(dist) ? "alchemist-web/dist is older than alchemist-web/src" : "alchemist-web/dist is missing";
  if (ARGS.has("--no-build")) throw new Error(`${why}: build it (npm ci, then tsc -b and vite build in alchemist-web) or drop --no-build`);
  log(`${why}: building the web bundle`);
  const run = (args) => execFileSync(process.execPath, args, { cwd: WEB, stdio: "inherit" });
  if (!existsSync(join(WEB, "node_modules/vite"))) execFileSync("npm", ["ci"], { cwd: WEB, stdio: "inherit" });
  run(["node_modules/typescript/bin/tsc", "-b"]);
  run(["node_modules/vite/bin/vite.js", "build"]);
}

// ---------------------------------------------------------------------------------------------------
// Servers: the ALchemist API (serves the app) and the simulator page
// ---------------------------------------------------------------------------------------------------
function portInUse(port) {
  const probe = (host) => new Promise((res) => {
    const s = connect({ host, port });
    s.once("connect", () => { s.destroy(); res(true); });
    s.once("error", () => res(false));
  });
  return Promise.all([probe("127.0.0.1"), probe("::1")]).then((r) => r.some(Boolean));
}
function freePort() {
  return new Promise((res, rej) => {
    const s = createServer();
    s.once("error", rej);
    s.listen(0, "127.0.0.1", () => { const { port } = s.address(); s.close(() => res(port)); });
  });
}
async function startApi(work) {
  // Never 8000: other tools and test suites on this machine talk to localhost:8000 and would share (and
  // change) this server's sessions. A release bundle calls this server on its own origin; the 8000 origin is
  // allowed, and openApp reroutes it, only in case the bundle is a dev build.
  const port = Number(process.env.PORT) || await freePort();
  if (await portInUse(port)) throw new Error(`port ${port} is busy`);
  const origins = [`http://localhost:${port}`, `http://127.0.0.1:${port}`, BUILT_ORIGIN].join(",");
  const proc = spawn(PYTHON, ["-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", String(port), "--log-level", "warning"], {
    cwd: work, env: { ...process.env, PYTHONPATH: REPO, ALLOWED_ORIGINS: origins }, stdio: ["ignore", "pipe", "pipe"],
  });
  apiProc = proc; // known to cleanup() from here on, so a failed start or an interrupt still stops it
  let out = "";
  proc.stdout.on("data", (d) => (out += d));
  proc.stderr.on("data", (d) => (out += d));
  const end = Date.now() + 120000;
  while (Date.now() < end) {
    if (proc.exitCode !== null) throw new Error(`the API exited (${proc.exitCode}):\n${out}`);
    try { if ((await fetch(`http://127.0.0.1:${port}/health`)).ok) break; } catch { /* not up yet */ }
    await sleep(300);
  }
  if (Date.now() >= end) throw new Error(`the API did not answer /health within 120 s:\n${out}`);
  writeFileSync(join(work, "api.log"), "");
  proc.stdout.on("data", (d) => writeFileSync(join(work, "api.log"), d, { flag: "a" }));
  proc.stderr.on("data", (d) => writeFileSync(join(work, "api.log"), d, { flag: "a" }));
  log(`API on port ${port} (working directory ${work})`);
  return { proc, port, origin: `http://localhost:${port}`, apiLog: () => out };
}
let apiProc = null;
async function startSimServer() {
  const html = readFileSync(join(DOCS, "tutorial/reactor/index.html"));
  const srv = createHttpServer((req, res) => { res.writeHead(200, { "content-type": "text/html; charset=utf-8" }); res.end(html); });
  await new Promise((r) => srv.listen(0, "127.0.0.1", r));
  return { srv, url: `http://localhost:${srv.address().port}/` };
}

// ---------------------------------------------------------------------------------------------------
// Page helpers
// ---------------------------------------------------------------------------------------------------
let page; // the ALchemist tab in use
const left = () => page.locator("div.w-\\[580px\\]");
const right = () => page.locator("div.w-\\[320px\\]");
const center = () => page.locator("div.flex-1.flex.flex-col.bg-background").first();
const modal = () => page.locator("div.fixed.inset-0 > div").first();

async function typeInto(loc, text) {
  await loc.click();
  await loc.press("ControlOrMeta+a");
  await loc.pressSequentially(String(text));
  const v = await loc.inputValue();
  if (v !== String(text)) throw new Error(`typed ${text}, field shows ${v}`);
}
async function paste(loc, text) {
  await loc.evaluate((el, t) => {
    const dt = new DataTransfer(); dt.setData("text/plain", t);
    el.dispatchEvent(new ClipboardEvent("paste", { clipboardData: dt, bubbles: true, cancelable: true }));
  }, text);
}
const toastsSeen = [];
/** Close every toast (they sit bottom-right and would end up in the shot) and wait until none is left. */
async function clearToasts(maxMs = 12000) {
  const end = Date.now() + maxMs;
  while (Date.now() < end) {
    const toasts = page.locator("[data-sonner-toast]");
    const n = await toasts.count();
    if (!n) return;
    for (let i = 0; i < n; i++) {
      const t = toasts.nth(i);
      const text = (await t.innerText().catch(() => "")).replace(/\s+/g, " ").trim();
      const type = await t.getAttribute("data-type").catch(() => null);
      if (text && !toastsSeen.some((s) => s.text === text)) toastsSeen.push({ type, text, at: +((Date.now() - t0) / 1000).toFixed(1) });
    }
    await page.locator("[data-sonner-toast] [data-close-button]").first().click({ timeout: 1000 }).catch(() => {});
    await page.mouse.move(700, 450);
    await sleep(250);
  }
  throw new Error("toasts did not go away");
}
let debugN = 0;
async function debugShot(label) {
  if (!DEBUG) return;
  mkdirSync(join(WORK, "debug"), { recursive: true });
  await page.screenshot({ path: join(WORK, "debug", `${String(++debugN).padStart(2, "0")}-${label}.png`) });
}
async function assertLight() {
  const dark = await page.evaluate(() => document.documentElement.classList.contains("dark"));
  if (dark) throw new Error("the app is in the dark theme");
}
/**
 * Switch the app to "light" or "dark" and wait until the UI (and any Plotly figure) has re-rendered.
 * The app keeps its theme in localStorage "theme" and as class "dark" on <html> (hooks/useTheme.tsx), and the
 * moon/sun button in the header (title "Switch to dark|light mode") toggles it. The button is clicked through
 * the DOM so an open dialog or menu does not block it and is not dismissed by a pointer-down; the page's own
 * state (open dialog, scroll position, form values) is untouched. The monitor view has no header and no
 * button, so there the same two things the hook sets are set directly.
 */
async function setTheme(theme) {
  const wantDark = theme === "dark";
  const how = await page.evaluate((dark) => {
    if (document.documentElement.classList.contains("dark") === dark) return "already";
    const btn = document.querySelector('header button[title^="Switch to"]');
    if (btn) { btn.click(); return "button"; }
    document.documentElement.classList.toggle("dark", dark);
    try { localStorage.setItem("theme", dark ? "dark" : "light"); } catch { /* storage blocked */ }
    return "direct";
  }, wantDark);
  await page.waitForFunction((dark) => document.documentElement.classList.contains("dark") === dark, wantDark, { timeout: 5000 });
  // a Plotly figure keeps the colours it was drawn with until the app re-renders it: wait for its paper
  // background to be dark (or light) before the shot
  await page.waitForFunction((dark) => {
    const luminance = (colour) => {
      const probe = document.createElement("i");
      probe.style.color = colour;
      document.body.appendChild(probe);
      const [r, g, b] = getComputedStyle(probe).color.match(/[\d.]+/g).map(Number);
      probe.remove();
      return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
    };
    return [...document.querySelectorAll(".js-plotly-plot")].every((gd) => {
      const bg = gd._fullLayout?.paper_bgcolor;
      return !bg || (luminance(bg) < 0.5) === dark;
    });
  }, wantDark, { timeout: 8000 }).catch(() => {
    throw new Error(`a Plotly figure did not re-render in the ${theme} theme (the contour plot must follow the header toggle)`);
  });
  await sleep(700);
  return how;
}

// ---------- control-name assertions ----------
const controlResults = [];
function nameMatcher(c) {
  return c.pattern ? new RegExp(c.pattern) : c.name;
}
async function anyVisible(loc) {
  const n = await loc.count();
  for (let i = 0; i < Math.min(n, 30); i++) if (await loc.nth(i).isVisible()) return true;
  return false;
}
async function controlVisible(c) {
  const name = nameMatcher(c);
  const exact = !c.pattern;
  const kinds = c.kind ? [c.kind] : ["button", "text", "option", "label", "placeholder", "menuitem"];
  for (const k of kinds) {
    let loc;
    if (k === "button") loc = page.getByRole("button", { name, exact });
    else if (k === "menuitem") loc = page.getByRole("menuitem", { name, exact });
    else if (k === "text") loc = page.getByText(name, { exact });
    else if (k === "label") loc = page.getByLabel(name, { exact });
    else if (k === "placeholder") loc = page.getByPlaceholder(name, { exact });
    else if (k === "option") {
      // an <option> counts when its <select> is visible
      const sel = page.locator("select:visible");
      loc = sel.locator("option").filter({ hasText: c.pattern ? name : new RegExp(`^\\s*${c.name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\s*$`) });
      if (await loc.count()) return k;
      continue;
    } else throw new Error(`unknown control kind ${k}`);
    if (await anyVisible(loc)) return k;
  }
  return null;
}
async function assertControls(step) {
  const due = CONTROLS.filter((c) => c.step === step);
  for (const c of due) {
    const app = page;
    if (c.app === "simulator") page = sim;
    const how = await controlVisible(c).finally(() => { page = app; });
    controlResults.push({ name: c.name, step, found: !!how, as: how });
    if (!how) log(`  MISSING control "${c.name}" at step ${step}`);
  }
  if (due.length) log(`  step ${step}: ${due.filter((c) => controlResults.find((r) => r.name === c.name && r.step === step)?.found).length}/${due.length} controls visible`);
}

// ---------- shots ----------
const shotResults = [];
async function shoot(name, { before } = {}) {
  const s = SHOTS.find((x) => x.name === name);
  if (!s) throw new Error(`shot ${name} is not in shots.json`);
  await clearToasts();
  if (before) await before();
  if (s.selector) {
    const target = page.locator(s.selector).first();
    await target.waitFor({ state: "visible", timeout: 30000 });
    if (s.scroll !== false) await target.evaluate((el, block) => el.scrollIntoView({ block, behavior: "instant" }), s.block ?? "nearest");
  }
  // no hover highlights, focus rings or Plotly mode bar (it shows while the pointer is over a plot)
  await page.mouse.move(VIEWPORT.width / 2, 2);
  await page.evaluate(() => document.activeElement?.blur?.());
  await sleep(1200);
  await assertLight();
  const path = join(IMAGES, s.file ?? join("guide", `${name}.png`));
  mkdirSync(dirname(path), { recursive: true });
  await page.screenshot({ path, type: "png" });
  const kb = Math.round(statSync(path).size / 1024);
  shotResults.push({ name, file: relative(REPO, path), kb });
  log(`  shot ${relative(REPO, path)} (${kb} KB)`);

  // the same frame in the dark theme, then back to light so the walkthrough carries on unchanged
  const darkPath = path.replace(/\.png$/, "-dark.png");
  const how = await setTheme("dark");
  try {
    if (s.selector) await page.locator(s.selector).first().waitFor({ state: "visible", timeout: 10000 }); // a menu or dialog can close on a theme change
    await page.mouse.move(VIEWPORT.width / 2, 2);
    await page.evaluate(() => document.activeElement?.blur?.());
    await sleep(800);
    await page.screenshot({ path: darkPath, type: "png" });
  } finally {
    await setTheme("light");
  }
  await sleep(300);
  await assertLight();
  const darkKb = Math.round(statSync(darkPath).size / 1024);
  shotResults.push({ name: `${name}-dark`, file: relative(REPO, darkPath), kb: darkKb, switchedBy: how });
  log(`  shot ${relative(REPO, darkPath)} (${darkKb} KB)`);
}

// ---------------------------------------------------------------------------------------------------
// The simulator: run setpoints as one queue, read every output from its run log
// ---------------------------------------------------------------------------------------------------
let sim;
const simRuns = [];
let simQueueChecked = false;
async function simulate(rows, reason) {
  await sim.bringToFront();
  if (await sim.locator("#batch-results").isVisible().catch(() => false)) await sim.locator("#queue-clear").click();
  const csv = ["Temperature,Pressure,H2_CO2,WHSV", ...rows.map((r) => [r.Temperature, r.Pressure, r.H2_CO2, r.WHSV].join(","))].join("\n");
  const before = await sim.evaluate(() => JSON.parse(localStorage.getItem("alchemist-reactor-sim/v2")).log.length);
  await sim.locator("#in-temperatureC").click();
  await paste(sim.locator("#in-temperatureC"), csv);
  if (rows.length > 1) {
    // several rows become a queue (the first queue checks the controls the tutorial names)
    await sim.locator("#queue").waitFor({ state: "visible", timeout: 10000 });
    if (!simQueueChecked) await assertControls("simulator-queue");
    await sim.locator("#queue-run").click();
    await sim.locator("#batch-results").waitFor({ state: "visible", timeout: 180000 });
    if (!simQueueChecked) { await assertControls("simulator-batch"); simQueueChecked = true; }
  } else {
    // one row fills the setpoint fields
    await sim.getByText("Filled from paste").waitFor({ timeout: 10000 });
    await sim.locator("#run").click();
    await sim.waitForFunction((n) => JSON.parse(localStorage.getItem("alchemist-reactor-sim/v2")).log.length > n, before, { timeout: 60000 });
    await sim.waitForFunction(() => { const b = document.getElementById("run"); return b && !b.disabled && !b.classList.contains("is-running"); }, null, { timeout: 60000 });
  }
  const runs = await sim.evaluate((n) => JSON.parse(localStorage.getItem("alchemist-reactor-sim/v2")).log.slice(n), before);
  if (runs.length !== rows.length) throw new Error(`simulator logged ${runs.length} runs for ${rows.length} setpoints`);
  await page.bringToFront();
  runs.sort((a, b) => a.run - b.run);
  return runs.map((r, i) => {
    const rec = {
      run: r.run, reason,
      Temperature: r.sp.temperatureC, Pressure: r.sp.pressureBar, H2_CO2: r.sp.h2Ratio, WHSV: r.sp.whsv,
      yield_pct: +(r.out.meohYield * 100).toFixed(2),
      co2_conversion_pct: +(r.out.co2Conversion * 100).toFixed(2),
      meoh_selectivity_pct: +(r.out.meohSelectivity * 100).toFixed(2),
    };
    for (const k of ["Temperature", "Pressure", "H2_CO2", "WHSV"]) {
      if (Math.abs(rec[k] - Number(rows[i][k])) > 1e-6) throw new Error(`simulator run ${r.run} ${k} = ${rec[k]}, sent ${rows[i][k]}`);
    }
    simRuns.push(rec);
    return rec;
  });
}

// ---------------------------------------------------------------------------------------------------
// REST helpers (setup for the scenarios whose subject is not the setup itself)
// ---------------------------------------------------------------------------------------------------
let API;
async function rest(method, path, body) {
  const r = await fetch(`${API.origin}/api/v1${path}`, { method, headers: { "content-type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
  const text = await r.text();
  if (!r.ok) throw new Error(`${method} ${path}: HTTP ${r.status} ${text}`);
  return text ? JSON.parse(text) : null;
}
async function stagedPoints(sessionId) {
  const r = await rest("GET", `/sessions/${sessionId}/experiments/staged`);
  return r.experiments ?? r.staged ?? r;
}

// ---------------------------------------------------------------------------------------------------
// App steps
// ---------------------------------------------------------------------------------------------------
async function openApp(context, query = "") {
  const p = await context.newPage();
  p.on("pageerror", (e) => pageErrors.push(e.message));
  if (API.port !== 8000) {
    // Not needed for a release bundle (it calls /api/v1 on this origin). A dev-mode bundle calls
    // http://localhost:8000/api/v1, so send those requests to this API instead.
    await p.route(/^http:\/\/localhost:8000\//, (route) => route.continue({ url: route.request().url().replace("localhost:8000", `localhost:${API.port}`) }));
  }
  await p.goto(`${API.origin}/${query}`);
  // a new page offers the last session's 30 s recovery backup; these sessions live on the server, so discard it
  const banner = p.getByText("Unsaved work detected");
  await banner.waitFor({ timeout: 2500 }).catch(() => {});
  if (await banner.isVisible()) { await p.getByRole("button", { name: "Discard" }).click(); await banner.waitFor({ state: "detached" }); }
  return p;
}
const pageErrors = [];

async function newSession(name, description) {
  await page.getByRole("button", { name: "New Session" }).click();
  await page.locator('input[placeholder="My Optimization Session"]').waitFor();
  await typeInto(page.locator('input[placeholder="My Optimization Session"]'), name);
  await typeInto(page.locator('textarea[placeholder^="Enter a brief"]'), description);
  await modal().getByRole("button", { name: "Save", exact: true }).click();
  await page.locator("text=Session Metadata").waitFor({ state: "detached" });
  const id = await page.evaluate(() => localStorage.getItem("alchemist_session_id"));
  if (!id) throw new Error("no session id in localStorage after New Session");
  return id;
}
async function addVariable(v, i, { beforeSubmit } = {}) {
  await page.getByRole("button", { name: "Add Variable" }).first().click();
  await page.locator('input[placeholder="e.g., temperature"]').waitFor();
  await typeInto(page.locator('input[placeholder="e.g., temperature"]'), v.name);
  await typeInto(page.locator('input[placeholder="Min value"]'), v.min);
  await typeInto(page.locator('input[placeholder="Max value"]'), v.max);
  if (v.unit) await typeInto(page.locator('input[placeholder^="e.g., °C"]'), v.unit);
  if (v.description) await typeInto(page.locator('input[placeholder="Brief description of this variable"], textarea[placeholder="Brief description of this variable"]').first(), v.description);
  if (v.changes) await page.getByLabel("Changes", { exact: true }).selectOption(v.changes);
  if (beforeSubmit) await beforeSubmit();
  await page.locator("form button[type=submit]").click();
  await page.locator('input[placeholder="e.g., temperature"]').waitFor({ state: "detached" });
  await page.getByText(`${i + 1} variable${i ? "s" : ""} defined`).waitFor();
}
/** The Add Variable form: the fields of each variable type, then back to Real so the form can be submitted. */
async function assertVariableForm() {
  await assertControls("variable-form");
  const type = page.getByLabel("Type", { exact: true });
  await type.selectOption("categorical");
  await assertControls("variable-categorical");
  await type.selectOption("discrete_numeric");
  await assertControls("variable-discrete");
  await type.selectOption("continuous");
}
/** The Initial Design panel's method-specific fields; leaves the method on LHS and the generated table in place. */
async function assertDesignMethods() {
  const method = left().locator("select").filter({ hasText: "Sobol" }).first();
  for (const [value, step] of [
    ["full_factorial", "initial-design-full-factorial"],
    ["fractional_factorial", "initial-design-fractional"],
    ["ccd", "initial-design-ccd"],
    ["gsd", "initial-design-gsd"],
  ]) {
    await method.selectOption(value);
    await assertControls(step);
  }
  await method.selectOption("lhs");
}
/** Open Add Point, fill the outputs (table paste for a batch), tick or untick the retrain box, save. */
async function recordStaged(outputs, { retrain, beforeSave, singleStep } = {}) {
  await page.getByRole("button", { name: /^Add Point/ }).click();
  const dlg = page.locator('[aria-label="Add experimental results"]');
  await dlg.waitFor({ timeout: 15000 });
  if (singleStep) await assertControls(singleStep);
  if (outputs.length > 1) {
    await dlg.getByRole("button", { name: "Table", exact: true }).click();
    const first = dlg.locator('[aria-label="Point 1 output"]');
    await first.click();
    await paste(first, outputs.map((y) => y.toFixed(2)).join("\n"));
    await sleep(500);
    const last = dlg.locator(`[aria-label="Point ${outputs.length} output"]`);
    if (Number(await last.inputValue()) !== Number(outputs.at(-1).toFixed(2))) throw new Error("table paste did not fill the Output column");
  } else {
    await typeInto(dlg.locator("#add-point-output"), outputs[0].toFixed(2));
  }
  const box = dlg.getByLabel("Retrain model (once, after saving)");
  if ((await box.isChecked()) !== !!retrain) await box.click();
  if (beforeSave) await beforeSave(dlg);
  const saved = page.waitForResponse((r) => r.request().method() === "POST" && r.url().includes("/experiments/record-batch"), { timeout: 300000 });
  await dlg.getByRole("button", { name: new RegExp(`^Save ${outputs.length} & close$`) }).click();
  const resp = await saved;
  if (!resp.ok()) throw new Error(`record-batch failed: HTTP ${resp.status()}`);
  await dlg.waitFor({ state: "detached", timeout: 120000 });
  // the retrain runs inside record-batch; give the panels a moment to refetch
  await sleep(retrain ? 2500 : 800);
}
/** Train BoTorch with the panel's defaults (Matern 2.5, normalize inputs, standardize outputs). */
let sklearnPanelChecked = false;
async function trainBoTorch() {
  await right().getByRole("button", { name: "Model", exact: true }).click();
  // the Model tab opens on scikit-learn: check its own controls once, before switching
  if (!sklearnPanelChecked) { await assertControls("train-sklearn"); sklearnPanelChecked = true; }
  await right().getByRole("button", { name: "BoTorch" }).click();
  // the defaults are the documented behaviour: check them, change nothing
  const selected = await right().locator("select").evaluateAll((els) => els.map((e) => e.value));
  for (const want of ["Matern", "2.5", "normalize", "standardize"]) {
    if (!selected.includes(want)) throw new Error(`BoTorch panel default ${want} not selected (selects show ${selected.join(", ")})`);
  }
  const trained = page.waitForResponse((r) => r.url().includes("/model/train") && r.request().method() === "POST", { timeout: 300000 });
  await right().getByRole("button", { name: "Train Model" }).click();
  const resp = await trained;
  if (!resp.ok()) throw new Error(`training failed: HTTP ${resp.status()}`);
  await right().getByText("Model trained successfully").waitFor({ timeout: 60000 }).catch(() => {});
  await right().getByText("Trained Model Info").waitFor({ timeout: 60000 });
}
function parseSuggestions(text, names, n) {
  const out = [];
  for (let i = 1; i <= n; i++) {
    const at = text.indexOf(`Suggestion ${i}:`);
    if (at < 0) break;
    const next = text.indexOf(`Suggestion ${i + 1}:`, at + 1);
    const block = text.slice(at, next > 0 ? next : undefined);
    out.push(Object.fromEntries(names.map((k) => [k, Number(block.match(new RegExp(`${k}:\\s*([-0-9.eE]+)`))[1])])));
  }
  return out;
}
const acqState = { mode: null, maximize: false };
async function acquire({ q = 1, stage = true } = {}) {
  await right().getByRole("button", { name: "Acquisition", exact: true }).click();
  const mode = q > 1 ? `qEI-${q}` : "logEI";
  if (acqState.mode !== mode) {
    if (q > 1) {
      await right().getByRole("button", { name: "Batch", exact: true }).click();
      await right().locator("select").filter({ hasText: "q-Expected Improvement" }).selectOption("qEI");
      await right().locator("select").filter({ hasText: /^\s*2\s*3\s*4/ }).selectOption(String(q));
    } else {
      await right().getByRole("button", { name: "Regular", exact: true }).click();
      await right().locator("select").filter({ hasText: "Log Expected Improvement" }).selectOption("logEI");
      await assertControls("acquisition-regular");
    }
    acqState.mode = mode;
  }
  if (!acqState.maximize) { await right().getByRole("button", { name: "Maximize" }).first().click(); acqState.maximize = true; }
  const suggested = page.waitForResponse((r) => r.url().includes("/acquisition/suggest") && r.request().method() === "POST", { timeout: 300000 });
  await right().getByRole("button", { name: "Run Acquisition Strategy" }).click();
  const resp = await suggested;
  if (!resp.ok()) throw new Error(`acquisition failed: HTTP ${resp.status()}`);
  await right().getByText(`Suggestion ${q}:`).waitFor({ timeout: 60000 });
  await sleep(800);
  const sugg = parseSuggestions(await right().evaluate((el) => el.innerText), REACTOR_VARIABLES.map((v) => v.name), q);
  if (sugg.length !== q) throw new Error(`expected ${q} suggestions, read ${sugg.length}`);
  if (stage) {
    await right().getByRole("button", { name: /Stage Suggestions/ }).click();
    await page.getByRole("button", { name: new RegExp(`^Add Point.*\\(${q}\\)`) }).waitFor({ timeout: 30000 });
  }
  return sugg;
}
async function showVisualizations() {
  if (!(await center().getByRole("button", { name: "Plot Contour", exact: true }).count())) {
    await right().getByRole("button", { name: "Model", exact: true }).click();
    await right().getByRole("button", { name: "Show Visualizations" }).click();
  }
  await center().getByRole("button", { name: "Plot Contour", exact: true }).waitFor({ timeout: 60000 });
}
/** Drag a contour fixed-value slider to its min or max (the slider commits on mouse-up). */
async function dragSlider(varName, toMax) {
  const box = center().getByLabel(`${varName} (slider)`, { exact: true });
  await box.scrollIntoViewIfNeeded();
  const b = await box.boundingBox();
  if (!b) throw new Error(`no slider for ${varName}`);
  const [v, lo, hi] = await box.evaluate((el) => [Number(el.value), Number(el.min), Number(el.max)]);
  const y = b.y + b.height / 2;
  const x0 = b.x + 8 + ((v - lo) / (hi - lo)) * (b.width - 16);
  const x1 = toMax ? b.x + b.width - 1 : b.x + 1;
  await page.mouse.move(x0, y);
  await page.mouse.down();
  for (let k = 1; k <= 12; k++) { await page.mouse.move(x0 + ((x1 - x0) * k) / 12, y); await sleep(20); }
  await page.mouse.up();
  await sleep(1200);
}
async function showContour() {
  await showVisualizations();
  await center().getByRole("button", { name: "Plot Contour", exact: true }).click();
  await sleep(1500);
  const sels = center().locator("select");
  for (let i = 0; i < (await sels.count()); i++) {
    const opts = await sels.nth(i).locator("option").allTextContents();
    if (opts.join("|") === "Temperature|Pressure|H2_CO2|WHSV") {
      const label = await sels.nth(i).evaluate((el) => el.closest("div")?.innerText ?? "");
      if (/X-Axis/.test(label) && (await sels.nth(i).inputValue()) !== "Temperature") { await sels.nth(i).selectOption("Temperature"); await sleep(800); }
      if (/Y-Axis/.test(label) && (await sels.nth(i).inputValue()) !== "Pressure") { await sels.nth(i).selectOption("Pressure"); await sleep(800); }
    }
  }
  await dragSlider("WHSV", false);
  await dragSlider("H2_CO2", true);
  const pts = center().getByLabel("Show Experimental Points");
  if (!(await pts.isChecked())) await pts.click();
  const nxt = center().getByLabel("Show Next Point");
  if ((await nxt.count()) && !(await nxt.isChecked())) await nxt.click();
  await sleep(2500);
  await center().locator(".js-plotly-plot").first().evaluate((el) => el.scrollIntoView({ block: "start", behavior: "instant" }));
  await sleep(1000);
}

// ---------------------------------------------------------------------------------------------------
// Scenario 1: the tutorial loop on the reactor simulator
// ---------------------------------------------------------------------------------------------------
const record = { simSeed: SIM_SEED, lhsSeed: LHS_SEED, simulator: "", scenarios: {} };
async function scenarioReactor(context) {
  log("scenario 1: the methanol reactor loop");
  page = await openApp(context);
  await page.getByRole("button", { name: "New Session" }).waitFor();
  await assertLight();
  await assertControls("welcome");
  const sessionId = await newSession("Methanol reactor", "CO2 hydrogenation to methanol on the ALchemist reactor simulator; maximize methanol yield (%).");
  record.scenarios.reactor = { sessionId };

  for (const [i, v] of REACTOR_VARIABLES.entries()) {
    await addVariable(v, i, i === REACTOR_VARIABLES.length - 1 ? { beforeSubmit: assertVariableForm } : {});
  }
  await assertControls("variables");
  await debugShot("variables");
  await shoot("variables");

  // initial design: LHS, 10 points, fixed seed
  const method = left().locator("select").filter({ hasText: "Sobol" }).first();
  await method.selectOption({ label: "LHS" });
  await typeInto(page.locator('input[type=number][min="5"]'), DOE_POINTS);
  await typeInto(page.locator('input[placeholder="Auto"]'), LHS_SEED);
  await page.getByRole("button", { name: "Generate Design" }).click();
  await page.getByText(`${DOE_POINTS} points generated`, { exact: true }).waitFor();
  await assertControls("initial-design");
  await shoot("initial-design");
  await assertDesignMethods();
  await page.getByRole("button", { name: "Stage", exact: true }).click();
  await page.getByRole("button", { name: new RegExp(`^Add Point.*\\(${DOE_POINTS}\\)`) }).waitFor({ timeout: 30000 });
  const design = await stagedPoints(sessionId);
  if (design.length !== DOE_POINTS) throw new Error(`staged ${design.length} points, expected ${DOE_POINTS}`);

  const doe = await simulate(design, "Initial Design");
  await recordStaged(doe.map((r) => r.yield_pct), {
    retrain: false,
    singleStep: "add-point-single",
    beforeSave: async () => { await assertControls("add-point"); await shoot("run-and-record"); },
  });
  await page.getByText(`${DOE_POINTS} rows`, { exact: true }).waitFor().catch(() => {});
  await assertControls("experiments");

  await trainBoTorch();
  await assertControls("train");
  await shoot("train");

  await right().getByRole("button", { name: "Show Visualizations" }).click();
  await center().getByText("Cross-Validation Parity Plot").waitFor({ timeout: 60000 });
  await sleep(2500);
  await assertControls("visualizations");
  await shoot("check-model");

  // one qEI batch, then logEI rounds, every point run on the simulator
  const batch = await acquire({ q: BATCH_Q });
  await assertControls("acquisition");
  await shoot("suggest");
  const batchRuns = await simulate(batch, "qEI batch");
  await recordStaged(batchRuns.map((r) => r.yield_pct), { retrain: true });
  for (let r = 1; r <= LOGEI_ROUNDS; r++) {
    const [s] = await acquire();
    const [run] = await simulate([s], "logEI");
    await recordStaged([run.yield_pct], { retrain: true });
  }
  // a fresh suggestion for the contour's "next point"
  await acquire({ stage: false });
  await showContour();
  await assertControls("contour");
  // hero: variables and data on the left, the contour in the centre, the model settings on the right
  await right().getByRole("button", { name: "Model", exact: true }).click();
  await sleep(800);
  await shoot("alchemist-web", {
    before: async () => {
      await left().evaluate((el) => el.scrollTo({ top: 0, behavior: "instant" }));
      await right().locator("text=Gaussian Process Model").first().evaluate((el) => el.scrollIntoView({ block: "start", behavior: "instant" })).catch(() => {});
    },
  });
  await right().getByRole("button", { name: "Acquisition", exact: true }).click();

  // header: the Save menu
  await page.locator("header").getByRole("button", { name: "Save options" }).click();
  await page.getByRole("menu").waitFor();
  await assertControls("header");
  await shoot("sessions");
  await page.keyboard.press("Escape");
  await page.getByRole("menu").waitFor({ state: "detached" }).catch(() => {});

  // optimal design panel on the same variables
  const oed = page.getByText("Optimal Design (OED)").first();
  await oed.scrollIntoViewIfNeeded();
  await oed.click().catch(() => {});
  await sleep(500);
  await assertControls("optimal-design-form");
  // The AI suggestion section is only on screen in Custom Effects mode. Open each state, check it, put it back
  // (nothing is generated, sent or saved, and the panel is in Quick mode again before the design is generated).
  await page.getByRole("button", { name: "Custom Effects", exact: true }).click();
  const aiToggle = page.getByRole("button", { name: /^Suggest effects with AI/ });
  await aiToggle.click();
  await assertControls("optimal-design-ai");
  await page.getByText("Ollama (local)", { exact: true }).click();
  await assertControls("optimal-design-ollama");
  await page.getByText("OpenAI", { exact: true }).click();
  await page.getByText("Edison Scientific", { exact: true }).click();
  await assertControls("optimal-design-edison");
  await page.getByText("Edison Scientific", { exact: true }).click();
  await aiToggle.click();
  await page.getByRole("button", { name: "Quick", exact: true }).click();
  const gen = page.getByRole("button", { name: "Generate Optimal Design" });
  await gen.scrollIntoViewIfNeeded();
  await gen.click();
  await page.getByText(/Design Quality/i).first().waitFor({ timeout: 120000 });
  await sleep(1000);
  await assertControls("optimal-design");
  await shoot("optimal-design");

  // monitoring dashboard for this session
  const mon = await openApp(context, `?session=${sessionId}&mode=monitor`);
  const main = page; page = mon;
  await sleep(4000);
  await assertLight();
  await assertControls("monitor");
  await shoot("monitoring");
  // after the shot: a controller reports "paused", and the Pause button becomes Resume
  await rest("PUT", `/sessions/${sessionId}/control`, { reported: "paused", reported_by: "docs", detail: "screenshot script" });
  await mon.getByRole("button", { name: "Resume", exact: true }).waitFor({ timeout: 15000 });
  await assertControls("monitor-paused");
  page = main;
  await mon.close();

  // last: the scikit-learn Acquisition panel needs a scikit-learn model (no shot is taken after this)
  await right().getByRole("button", { name: "Model", exact: true }).click();
  await right().getByRole("button", { name: "scikit-learn", exact: true }).click();
  const skTrained = page.waitForResponse((r) => r.url().includes("/model/train") && r.request().method() === "POST", { timeout: 300000 });
  await right().getByRole("button", { name: "Train Model" }).click();
  if (!(await skTrained).ok()) throw new Error("scikit-learn training failed");
  await right().getByText("Backend:").locator("..").getByText("sklearn", { exact: true }).waitFor({ timeout: 60000 });
  await right().getByRole("button", { name: "Acquisition", exact: true }).click();
  await right().getByText("Acquisition Strategy", { exact: true }).waitFor({ timeout: 30000 });
  await assertControls("acquisition-sklearn");

  record.scenarios.reactor.experiments = simRuns.length;
  return sessionId;
}

// ---------------------------------------------------------------------------------------------------
// Scenario 2: multi-objective (two simulator outputs, loaded from a CSV)
// ---------------------------------------------------------------------------------------------------
async function scenarioMultiObjective(context, work) {
  log("scenario 2: multi-objective from a CSV");
  const runs = simRuns.filter((r) => r.reason === "Initial Design");
  const csv = join(work, "reactor_two_objectives.csv");
  writeFileSync(csv, ["Temperature,Pressure,H2_CO2,WHSV,CO2_conversion,MeOH_selectivity",
    ...runs.map((r) => [r.Temperature, r.Pressure, r.H2_CO2, r.WHSV, r.co2_conversion_pct, r.meoh_selectivity_pct].join(","))].join("\n") + "\n");
  page = await openApp(context);
  await page.getByRole("button", { name: "New Session" }).waitFor();
  if (await page.locator("header").getByRole("button", { name: "Clear" }).count()) await page.locator("header").getByRole("button", { name: "Clear" }).click();
  await newSession("Methanol reactor, two objectives", "CO2 conversion and methanol selectivity from the reactor simulator.");
  for (const [i, v] of REACTOR_VARIABLES.entries()) await addVariable(v, i);
  const fc = page.waitForEvent("filechooser");
  await page.getByRole("button", { name: "Load CSV" }).click();
  await (await fc).setFiles(csv);
  await page.getByText("Select Target Column(s)").waitFor();
  await page.getByText("Multi-Objective", { exact: true }).click();
  for (const col of ["CO2_conversion", "MeOH_selectivity"]) await modal().getByLabel(col).check().catch(async () => { await modal().getByText(col, { exact: true }).click(); });
  await assertControls("target-columns");
  await shoot("multi-objective");
  await page.getByRole("button", { name: "Confirm" }).click();
  await page.getByText(`${runs.length} rows`, { exact: true }).waitFor().catch(() => {});
  await page.close();
}

// ---------------------------------------------------------------------------------------------------
// Scenario 3: split-plot (a furnace temperature that is hard to change), as in guide/more/split-plot.md
// ---------------------------------------------------------------------------------------------------
const SP_VARIABLES = [
  { name: "furnace_temp", min: 600, max: 900, unit: "°C", description: "Furnace setpoint", changes: "hard" },
  { name: "flow", min: 1, max: 5, unit: "L/min", description: "Gas flow" },
  { name: "time", min: 10, max: 60, unit: "min", description: "Hold time" },
];
// the page's own synthetic response (guide/more/split-plot.md, "Run the design in whole-plot order")
const spResponse = (p) => -(((p.furnace_temp - 780) / 100) ** 2) - ((p.flow - 3) / 2) ** 2 - ((p.time - 35) / 25) ** 2;
async function scenarioSplitPlot(context) {
  log("scenario 3: split-plot");
  page = await openApp(context);
  await page.getByRole("button", { name: "New Session" }).waitFor();
  const sessionId = await newSession("Furnace split-plot", "Synthetic response; furnace temperature is hard to change.");
  for (const [i, v] of SP_VARIABLES.entries()) await addVariable(v, i);
  // reopen the hard variable: the form shows its Changes level, the list behind it the hard badge
  const row = page.locator("div.grid").filter({ hasText: "furnace_temp" }).filter({ has: page.getByRole("button", { name: "Edit", exact: true }) }).last();
  await row.getByRole("button", { name: "Edit", exact: true }).click();
  await page.getByText("Edit Variable").first().waitFor();
  await assertControls("sp-variable-form");
  await shoot("split-plot-variables");
  await page.getByRole("button", { name: "Cancel" }).click();

  // split-plot design: hard = 2-level full factorial, replicated twice; easy = 4 LHS points; reference run at start
  await page.getByRole("button", { name: "Split-plot", exact: true }).click();
  await page.getByLabel("hard method", { exact: true }).selectOption("full_factorial");
  await typeInto(page.getByLabel("hard levels", { exact: true }), 2);
  await typeInto(page.getByLabel("hard replicates", { exact: true }), 2);
  await page.getByLabel("easy method", { exact: true }).selectOption("lhs");
  await typeInto(page.getByLabel("easy points", { exact: true }), 4);
  await page.getByLabel("reference runs", { exact: true }).selectOption("start");
  // seed 1: the four whole plots alternate 600/900 °C, which reads more clearly than two equal plots in a row
  await typeInto(page.getByLabel("split-plot seed", { exact: true }), 1);
  await page.getByRole("button", { name: "Generate split-plot design" }).click();
  await page.getByText(/whole plots × \d+ runs = \d+ runs/).waitFor({ timeout: 60000 });
  await assertControls("sp-design");
  await shoot("split-plot-design");

  // stage, record the page's synthetic response, train, and open the Hold list
  await page.getByRole("button", { name: "Stage", exact: true }).click();
  await page.getByRole("button", { name: /^Add Point.*\(\d+\)/ }).waitFor({ timeout: 30000 });
  const staged = await stagedPoints(sessionId);
  await recordStaged(staged.map((p) => spResponse(p)), { retrain: false, beforeSave: () => assertControls("sp-add-point") });
  await trainBoTorch();
  await right().getByRole("button", { name: "Acquisition", exact: true }).click();
  await page.getByText("Hold", { exact: true }).waitFor({ timeout: 30000 });
  await assertControls("sp-acquisition");
  await page.close();
  record.scenarios.splitPlot = { sessionId, runs: staged.length };
}

// Scenario 4: a very hard and an observed variable (set up over REST, opened with Join): the menus they add
async function scenarioSplitPlotLevels(context) {
  log("scenario 4: very hard and observed variables");
  const { session_id: sid } = await rest("POST", "/sessions", {});
  for (const v of [
    { name: "furnace_temp", type: "real", min: 600, max: 900, changes: "very_hard" },
    { name: "flow", type: "real", min: 1, max: 5, changes: "hard" },
    { name: "time", type: "real", min: 10, max: 60 },
    { name: "ambient", type: "real", min: 15, max: 30, role: "observed" },
  ]) await rest("POST", `/sessions/${sid}/variables`, v);
  const rows = [];
  for (const [i, temp] of [600, 700, 800, 900].entries()) {
    for (let k = 0; k < 3; k++) {
      const flow = 1 + ((i * 3 + k) * 0.37) % 4, time = 10 + ((i * 7 + k * 13) % 50), ambient = 18 + ((i + k * 2) % 8);
      rows.push({ inputs: { furnace_temp: temp, flow, time, ambient }, output: spResponse({ furnace_temp: temp, flow, time }) + 0.05 * ambient });
    }
  }
  await rest("POST", `/sessions/${sid}/experiments/batch`, { experiments: rows });
  await rest("POST", `/sessions/${sid}/model/train`, { backend: "botorch" });
  page = await openApp(context, `?session=${sid}`);
  await page.getByText("Variable Management").first().waitFor({ timeout: 30000 });
  await page.getByRole("button", { name: "Split-plot", exact: true }).click();
  await assertControls("spl-design");
  await right().getByRole("button", { name: "Acquisition", exact: true }).click();
  await page.getByText("Measured now", { exact: true }).waitFor({ timeout: 30000 });
  await assertControls("spl-acquisition");
  await page.close();
  record.scenarios.splitPlotLevels = { sessionId: sid, experiments: rows.length };
}

// ---------------------------------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------------------------------
// always a new folder: the API keeps sessions and recovery backups in cache/ under its working directory
if (process.env.WORK) mkdirSync(process.env.WORK, { recursive: true });
const WORK = mkdtempSync(join(process.env.WORK ?? tmpdir(), "alchemist-shots-"));
let browser, simSrv, ok = false;

/** Stop everything this script started. Synchronous so the signal handlers can call it. */
function cleanup() {
  const bp = browser?.process?.();
  if (bp?.pid && bp.exitCode === null) {
    try { process.kill(-bp.pid, "SIGKILL"); } catch { try { bp.kill("SIGKILL"); } catch { /* gone */ } } // Playwright starts Chromium in its own process group
  }
  try { simSrv?.close(); } catch { /* not listening */ }
  if (apiProc && apiProc.exitCode === null) { try { apiProc.kill("SIGKILL"); } catch { /* gone */ } }
}
for (const sig of ["SIGINT", "SIGTERM"]) {
  process.once(sig, () => {
    console.error(`\n${sig}: stopping the API server and the browser`);
    cleanup();
    process.exit(130);
  });
}

try {
  if (!checkPages()) throw new Error("fix controls.json first (see above)");
  ensureBundle();
  API = await startApi(WORK);
  const simServer = await startSimServer();
  simSrv = simServer.srv;
  record.simulator = readFileSync(join(DOCS, "tutorial/reactor/SOURCE.md"), "utf8").match(/commit (\w+)/)?.[1] ?? "unknown";

  const { chromium } = await import("playwright");
  // our own SIGINT/SIGTERM handlers stop the browser (see cleanup)
  browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME || undefined, handleSIGINT: false, handleSIGTERM: false, handleSIGHUP: false });
  const context = await browser.newContext({ viewport: VIEWPORT, deviceScaleFactor: 1, colorScheme: "light", acceptDownloads: true });
  await context.addInitScript(() => { try { localStorage.setItem("theme", "light"); } catch { /* storage blocked */ } });

  sim = await context.newPage();
  await sim.goto(simServer.url);
  await sim.evaluate((seed) => localStorage.setItem("alchemist-reactor-sim/v2", JSON.stringify({ log: [], nextRun: 1, sessionSeed: seed, noiseOn: true, theme: "light", lastSp: { temperatureC: 250, pressureBar: 50, h2Ratio: 3, whsv: 4 } })), SIM_SEED);
  await sim.reload();
  await sim.locator("#run").waitFor();
  await assertControls("simulator");

  await scenarioReactor(context);
  await scenarioMultiObjective(context, WORK);
  await scenarioSplitPlot(context);
  await scenarioSplitPlotLevels(context);
  ok = true;
} catch (e) {
  console.error(e);
  if (page) await page.screenshot({ path: join(WORK, "error.png") }).catch(() => {});
  if (sim) await sim.screenshot({ path: join(WORK, "error-simulator.png") }).catch(() => {});
  console.error(`error screenshot and API log in ${WORK}`);
} finally {
  if (browser) await browser.close().catch(() => {});
  if (apiProc && apiProc.exitCode === null) { apiProc.kill("SIGINT"); await sleep(1500); }
  cleanup(); // whatever is still running
}

// results
const missingControls = controlResults.filter((r) => !r.found);
const neverChecked = CONTROLS.filter((c) => !controlResults.some((r) => r.name === c.name && r.step === c.step));
const missingShots = SHOTS.flatMap((s) => [s.name, `${s.name}-dark`]).filter((n) => !shotResults.some((r) => r.name === n)).map((name) => ({ name }));
const big = shotResults.filter((s) => s.kb > 500);
if (ok) {
  writeFileSync(join(HERE, "simulated-session.json"), JSON.stringify({
    _about: "Every simulator run behind the docs screenshots, written by shoot.mjs. Simulated data (docs/tutorial/reactor), never lab data.",
    simulatorCommit: record.simulator, simSeed: SIM_SEED, lhsSeed: LHS_SEED, noise: "2 % relative (simulator default)",
    variables: REACTOR_VARIABLES, runs: simRuns,
  }, null, 2) + "\n");
}
writeFileSync(join(WORK, "result.json"), JSON.stringify({ ok, shots: shotResults, controls: controlResults, toasts: toastsSeen, pageErrors, scenarios: record.scenarios }, null, 2));
console.log(`\n${shotResults.length}/${SHOTS.length * 2} shots (light and dark), ${controlResults.filter((r) => r.found).length}/${controlResults.length} control checks passed. Details: ${join(WORK, "result.json")}`);
for (const s of missingShots) console.error(`  shot not taken: ${s.name}`);
for (const c of missingControls) console.error(`  control not visible: "${c.name}" at step ${c.step}`);
for (const c of neverChecked) console.error(`  control never checked (step "${c.step}" not reached): "${c.name}"`);
for (const s of big) console.error(`  over 500 KB: ${s.file} (${s.kb} KB)`);
const errors = toastsSeen.filter((t) => t.type === "error");
for (const t of errors) console.error(`  error toast at ${t.at} s: ${t.text}`);
process.exit(ok && !missingControls.length && !neverChecked.length && !missingShots.length ? 0 : 1);
