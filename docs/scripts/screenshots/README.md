# Docs screenshots

`shoot.mjs` takes every screenshot the docs use from the real web app, on simulated data, at 1440x900 in the light theme. Run it before each release, and after any change to the web app that a guide page shows. It is not run in CI.

## Run

```bash
cd docs/scripts/screenshots
npm install            # once: Playwright, pinned in package.json
node shoot.mjs         # about 3 minutes
```

It needs Node 20 or later and a Python environment with ALchemist's dependencies (`~/miniforge3/envs/alchemist-env/bin/python` by default; set `ALCHEMIST_PYTHON` otherwise). Playwright 1.63.0 uses Chromium build 1243; if that build is not in `~/Library/Caches/ms-playwright`, install it with `node node_modules/playwright/cli.js install chromium`.

What it does:

1. Rebuilds the web bundle (`alchemist-web/dist`) if it is older than `alchemist-web/src` (`--no-build` fails instead).
2. Starts the API from this checkout on a free port (never 8000) with a new working directory, so no old session or recovery backup shows up.
3. Opens the reactor simulator embedded in the docs (`docs/tutorial/reactor/index.html`) in a second tab with a fixed noise seed, and runs every experiment there.
4. Walks the tutorial loop in the app: four simulator variables, an LHS design (10 points, seed 42), results by Add Point, a BoTorch model, a qEI batch of 4 and two logEI rounds, the contour plot. Then short scenarios for multi-objective, split-plot, and very hard and observed variables.
5. Writes the images in `shots.json` to `docs/images/` and every simulator run to `simulated-session.json`.

The run exits non-zero if a shot is missing or a control name is not visible; the API log and an error screenshot are left in the run's temp folder, which the last lines of output name. `--debug` saves extra screenshots there.

## Files

| File | What it holds |
|---|---|
| `shots.json` | Every screenshot: name, page that shows it, what must be on screen. |
| `controls.json` | Every control name the docs put in bold, and the step at which the script checks it is visible. |
| `simulated-session.json` | Output: each simulator run behind the images (setpoints, yield, conversion, selectivity). |

## When you write or change a page

- A new screenshot: add an entry to `shots.json` and a `shoot("<name>")` call at the right step of `shoot.mjs`.
- A new bold control name: add it to `controls.json` with the step where the app shows it. `node shoot.mjs --check-pages` (no browser, a second) fails when a page in `docs/guide/` or `docs/get-started/` bolds a name that is not in `controls.json`.
