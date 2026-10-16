<p align="center">
  <a href="https://natlabrockies.github.io/ALchemist/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/NatLabRockies/ALchemist/main/docs/assets/NEW_LOGO_DARK.png">
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/NatLabRockies/ALchemist/main/docs/assets/NEW_LOGO_LIGHT.png">
      <img src="https://raw.githubusercontent.com/NatLabRockies/ALchemist/main/docs/assets/NEW_LOGO_LIGHT.png" alt="ALchemist" width="380">
    </picture>
  </a>
</p>

<p align="center">
  <b>Active learning and Bayesian optimization for experimental research.</b><br>
  Define the variables, run a first design, record the experimental results, and let a Gaussian process choose the next experiment.
</p>

<p align="center">
  <a href="https://pypi.org/project/alchemist-nrel/"><img src="https://img.shields.io/pypi/v/alchemist-nrel.svg" alt="PyPI version"></a>
  <a href="https://github.com/NatLabRockies/ALchemist/actions/workflows/tests.yml"><img src="https://github.com/NatLabRockies/ALchemist/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://github.com/NatLabRockies/ALchemist/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" alt="License: BSD-3-Clause"></a>
</p>

<p align="center">
  <a href="https://natlabrockies.github.io/ALchemist/get-started/install/">Install</a>&nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <a href="https://natlabrockies.github.io/ALchemist/get-started/first-loop/">Your first loop</a>&nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <a href="https://natlabrockies.github.io/ALchemist/">Documentation</a>&nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <a href="https://natlabrockies.github.io/ALchemist/reference/changelog/">Changelog</a>
</p>

<p align="center">
  <a href="https://natlabrockies.github.io/ALchemist/get-started/first-loop/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/NatLabRockies/ALchemist/main/docs/images/alchemist-web-dark.png">
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/NatLabRockies/ALchemist/main/docs/images/alchemist-web.png">
      <img src="https://raw.githubusercontent.com/NatLabRockies/ALchemist/main/docs/images/alchemist-web-dark.png" alt="The ALchemist web app on a simulated reactor session: the variables and 16 results on the left, a contour plot of the model's predicted yield over temperature and pressure in the centre, and the Gaussian process model settings and cross-validation metrics on the right." width="100%">
    </picture>
  </a>
</p>


<table align="center">
<tr>
<td width="33%" valign="top"><h3 align="center"><a href="https://natlabrockies.github.io/ALchemist/guide/initial-design/">Design</a></h3>
<p align="center">Latin hypercube, Sobol, central composite and optimal designs, with linear constraints on your variables.</p></td>
<td width="33%" valign="top"><h3 align="center"><a href="https://natlabrockies.github.io/ALchemist/guide/check-model/">Model</a></h3>
<p align="center">Gaussian-process models, checked with parity, Q-Q and calibration plots from cross-validation.</p></td>
<td width="33%" valign="top"><h3 align="center"><a href="https://natlabrockies.github.io/ALchemist/guide/suggest/">Suggest</a></h3>
<p align="center">Expected improvement, upper confidence bound and batch acquisition. Stage the suggested points, then record the results once you have run them.</p></td>
</tr>
</table>

## Your first loop

Twenty minutes in the web app against a simulated reactor, without writing any code. The full walkthrough is [Your first loop](https://natlabrockies.github.io/ALchemist/get-started/first-loop/).

1. Define the variables, with their ranges and units.
2. Generate an initial design of ten Latin hypercube points and stage them.
3. Run them and paste the results in as one table.
4. Train a Gaussian process and check its parity and calibration plots.
5. Ask for the next experiment, one point or a batch of four, and repeat.

<table align="center">
<tr>
<td width="50%" align="center"><a href="https://natlabrockies.github.io/ALchemist/guide/check-model/"><picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/NatLabRockies/ALchemist/main/docs/images/readme/contour-after-design.png">
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/NatLabRockies/ALchemist/main/docs/images/readme/contour-after-design-light.png">
      <img src="https://raw.githubusercontent.com/NatLabRockies/ALchemist/main/docs/images/readme/contour-after-design.png" alt="Contour plot of predicted methanol yield over temperature and pressure after the ten design points: a broad maximum around 280 °C and 60 bar" width="100%">
    </picture></a><br><sub>After the ten-point design</sub></td>
<td width="50%" align="center"><a href="https://natlabrockies.github.io/ALchemist/guide/suggest/"><picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/NatLabRockies/ALchemist/main/docs/images/readme/contour-after-loop.png">
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/NatLabRockies/ALchemist/main/docs/images/readme/contour-after-loop-light.png">
      <img src="https://raw.githubusercontent.com/NatLabRockies/ALchemist/main/docs/images/readme/contour-after-loop.png" alt="The same contour after twelve suggested experiments: a sharp ridge at 80 bar peaking near 262 °C, with the suggested points clustered on it" width="100%">
    </picture></a><br><sub>After twelve suggested experiments</sub></td>
</tr>
</table>

## Install

```bash
pip install alchemist-nrel
alchemist-web
```

Python 3.11 or later; the first install downloads PyTorch and BoTorch. Open `http://localhost:8000` for the web app. The same engine is a Python library and a REST API:

```python
from alchemist_core import OptimizationSession

session = OptimizationSession()
session.add_variable("Temperature", "real", bounds=(200, 300), unit="°C")
session.add_variable("Pressure", "real", bounds=(20, 80), unit="bar")
# ... load results, then
session.train_model(backend="botorch")
session.suggest_next(strategy="LogEI")
```

One session file serves all three: the web app for a person in the loop, the Python API for notebooks and scripts, and the REST API, with a work queue, for a reactor or robot that runs the loop on its own. More in the [documentation](https://natlabrockies.github.io/ALchemist/).

<details>
<summary>Overview, architecture and use cases</summary>

## Overview

**Key features:**

- Real, integer, discrete and categorical variables, with linear input constraints that every design and suggestion respects
- Space-filling (LHS, Sobol), classical (CCD, Box-Behnken, factorial), screening (Plackett-Burman, GSD), D/A/I-optimal and split-plot designs
- AI-assisted choice of effects for optimal designs, through OpenAI, a local Ollama model or Edison Scientific literature search
- Gaussian-process models from BoTorch (Matern, RBF, IBNN kernels) or scikit-learn, with cross-validation diagnostics
- Expected Improvement (and its log form), Probability of Improvement, Upper Confidence Bound, batch qEI and qUCB, and qNIPV for exploration
- Several objectives at once, with Pareto fronts and hypervolume tracking
- A web app (React over FastAPI), a Python session API and a REST API over one session file, with an audit log
- A work queue that records what was suggested next to what was actually run, for reactor-in-the-loop and automated-laboratory work
- The deprecated CustomTkinter desktop app, installed with the `[desktop]` extra

**Architecture:**

ALchemist has four parts:

- **Core Session API**: Headless Bayesian optimization engine (`alchemist_core`) that powers all interfaces
- **Desktop Application (deprecated)**: CustomTkinter GUI using the Core Session API, designed for human-in-the-loop and offline optimization
- **REST API**: FastAPI server providing a thin wrapper around the Core Session API for remote access; includes a work queue for autonomous reactor-in-the-loop workflows
- **Web Application**: React UI consuming the REST API, supporting both interactive and autonomous optimization workflows; real-time WebSocket event streaming for live session monitoring

The web app, the Python API and the REST API share one session file format (JSON). The deprecated desktop app opens the same files, but refuses to save, train or log a session that holds something it cannot keep, such as an observed variable, input constraints or the provenance columns the work queue adds; see [Desktop app (legacy)](https://natlabrockies.github.io/ALchemist/reference/desktop/).

---

## Use Cases

- **Interactive Optimization**: Use the web app (or the deprecated desktop app) for manual experiment design and human-in-the-loop optimization
- **Programmatic Workflows**: Import the Session API in Python scripts or Jupyter notebooks for batch processing
- **Autonomous Optimization**: Use the REST API to integrate ALchemist with automated laboratory equipment in a closed loop (ALchemist never starts or stops a cycle; the controller does)
- **Remote Monitoring**: When a controller drives the session, the web app's monitor view shows the work queue and model health and lets you send pause and resume requests

For a worked example, see [Your first loop](https://natlabrockies.github.io/ALchemist/get-started/first-loop/) in the documentation.

---


</details>

<details>
<summary>Installing from source, the deprecated desktop app, development status</summary>

## Installation

**Requirements:** Python 3.11 or higher

**Recommended (Optional):** We recommend using [Anaconda](https://www.anaconda.com/products/distribution) to manage Python environments:

```bash
conda create -n alchemist-env python=3.13
conda activate alchemist-env
```

**Basic Installation:**

```bash
pip install alchemist-nrel
```
The desktop app is deprecated and is no longer installed by default; it needs `pip install "alchemist-nrel[desktop]"`.

**From GitHub:**
> *Note: This installs the latest unreleased version. The web application is not pre-built with this method because static build files are not included in the repository.*

```bash
pip install git+https://github.com/NatLabRockies/ALchemist.git
```

For advanced installation options, Docker deployment, and development setup, see the [Install page](https://natlabrockies.github.io/ALchemist/get-started/install/) in the documentation.

---

## Running ALchemist

**Web Application:**
```bash
alchemist-web
```
Opens at [http://localhost:8000](http://localhost:8000)

**Desktop Application (deprecated):**
```bash
pip install "alchemist-nrel[desktop]"
alchemist
```

For detailed usage instructions, see [Install](https://natlabrockies.github.io/ALchemist/get-started/install/) and [Your first loop](https://natlabrockies.github.io/ALchemist/get-started/first-loop/) in the documentation.

---

## Development Status

ALchemist is under active development at NLR as part of the DataHub project within the ChemCatBio consortium.

---

## Issues & Troubleshooting

If you encounter any issues or have questions, please [open an issue on GitHub](https://github.com/NatLabRockies/ALchemist/issues) or contact ccoatney@nlr.gov.

For the latest known issues and troubleshooting tips, see [Troubleshooting](https://natlabrockies.github.io/ALchemist/get-started/troubleshooting/) in the documentation.

---


</details>

## Cite and license

**NLR Software Record:** SWR-25-102

**Cite:** Coatney, C. *ALchemist: Active Learning Toolkit for Chemical and Materials Research*. National Laboratory of the Rockies, software record SWR-25-102, DOE CODE, [doi:10.11578/dc.20250707.1](https://doi.org/10.11578/dc.20250707.1).

BSD 3-Clause License; see [LICENSE](https://github.com/NatLabRockies/ALchemist/blob/main/LICENSE). Questions and bug reports: [GitHub Issues](https://github.com/NatLabRockies/ALchemist/issues).
