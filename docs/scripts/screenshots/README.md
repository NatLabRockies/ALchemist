# Docs screenshots

`shoot.mjs` takes every screenshot the docs use from the real web app, on simulated data, at 1440x900, each in the light and in the dark theme. Run it before each release, and after any change to the web app that a guide page shows. It is not run in CI.

## Run

```bash
cd docs/scripts/screenshots
npm install            # once: Playwright, pinned in package.json
node shoot.mjs         # about 3 minutes
```

It needs Node 20 or later and a Python environment with ALchemist's dependencies (`~/miniforge3/envs/alchemist-env/bin/python` by default, or `python3` on your PATH when that file does not exist; set `ALCHEMIST_PYTHON` to use another). Playwright 1.63.0 uses Chromium build 1243; if that build is not in `~/Library/Caches/ms-playwright`, install it with `node node_modules/playwright/cli.js install chromium`.

What it does:

1. Rebuilds the web bundle (`alchemist-web/dist`) if it is older than `alchemist-web/src` (`--no-build` fails instead).
2. Starts the API from this checkout on a free port (never 8000) with a new working directory, so no old session or recovery backup shows up.
3. Opens the reactor simulator embedded in the docs (`docs/tutorial/reactor/index.html`) in a second tab with a fixed noise seed, and runs every experiment there.
4. Walks the tutorial loop in the app: four simulator variables, an LHS design (10 points, seed 42), results by Add Point, a BoTorch model, a qEI batch of 4 and two logEI rounds, the contour plot. Then short scenarios for multi-objective, split-plot, and very hard and observed variables.
5. Writes the images in `shots.json` to `docs/images/` and every simulator run to `simulated-session.json`. Each shot is taken twice from the same frame: `<name>.png` in the light theme, then `<name>-dark.png` after switching the app to its dark theme (see below), then the app goes back to light so the walkthrough carries on unchanged.

### How the dark shot is taken

The app keeps its theme in `localStorage` key `theme` (`light` or `dark`; the app's default is dark, so the script sets `light` before the page loads) and as the class `dark` on `<html>` (`alchemist-web/src/hooks/useTheme.tsx`). The moon/sun button in the header (title "Switch to dark mode") toggles it. `setTheme()` in `shoot.mjs` clicks that button through the DOM, so an open dialog or menu does not get in the way, waits for the class, and waits until any Plotly figure has re-drawn its paper background in the new theme. The monitor view has no header, so there `setTheme()` sets the class and the `localStorage` key itself. If a figure does not follow the toggle, or an open menu closes on the switch, the run fails and says which.

The pages show the pair with `.only-light` and `.only-dark`: `![alt](x.png){ .only-light }` followed by `![alt](x-dark.png){ .only-dark }` (in `overrides/home.html`, `class="only-light"` and `class="only-dark"` on two `<img>` tags). The README's hero stays light, since PyPI cannot switch.

The run exits non-zero if a shot is missing or a control name is not visible; the API log and an error screenshot are left in the run's temp folder, which the last lines of output name. `--debug` saves extra screenshots there.

## Files

| File | What it holds |
|---|---|
| `shots.json` | Every screenshot: name, page that shows it, what must be on screen. |
| `controls.json` | Every control name the docs put in bold, and the step at which the script checks it is visible. |
| `simulated-session.json` | Output: each simulator run behind the images (setpoints, yield, conversion, selectivity). |

## When you write or change a page

- A new screenshot: add an entry to `shots.json` and a `shoot("<name>")` call at the right step of `shoot.mjs`. `shoot()` takes the dark twin itself. Show both on the page, as above.
- A new bold control name: add it to `controls.json` with the step where the app shows it. `node shoot.mjs --check-pages` (no browser, a second) fails when a page in `docs/guide/` or `docs/get-started/` bolds a name that is not in `controls.json`. It also fails when a page or `overrides/home.html` shows an app screenshot without its `-dark.png` beside it (light image `.only-light`, dark image `.only-dark`), or when a `-dark.png` file is missing from `docs/images/`. `tests/docs/test_screenshot_pairs.py` checks the same markup in CI.
