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
  Define the variables, run a first design, record what the lab measured, and let a Gaussian process choose the next experiment.
</p>

<p align="center">
  <a href="https://pypi.org/project/alchemist-nrel/"><img src="https://badge.fury.io/py/alchemist-nrel.svg" alt="PyPI version"></a>
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

<p align="center"><sub>The web app on a simulated methanol reactor: variables and results on the left, the model's predicted yield over temperature and pressure in the centre, the Gaussian process on the right. The screenshot follows your GitHub theme.</sub></p>

ALchemist is a modular Python toolkit that brings active learning and Bayesian optimization to experimental design in chemical and materials research. It is written for scientists and engineers who want to explore or optimize high-dimensional variable spaces with a graphical interface, a Python or REST API, or an autonomous optimization workflow.

**NLR Software Record:** SWR-25-102

**Cite:** Coatney, C. *ALchemist: Active Learning Toolkit for Chemical and Materials Research*. National Laboratory of the Rockies, software record SWR-25-102, DOE CODE, [doi:10.11578/dc.20250707.1](https://doi.org/10.11578/dc.20250707.1).

---

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

## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](https://github.com/NatLabRockies/ALchemist/blob/main/LICENSE) file for details.

---

## Repository

[https://github.com/NatLabRockies/ALchemist](https://github.com/NatLabRockies/ALchemist)

