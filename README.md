<img src="docs/assets/NEW_LOGO_LIGHT.png" alt="ALchemist" width="50%" />

**ALchemist: Active Learning Toolkit for Chemical and Materials Research**

[![PyPI version](https://badge.fury.io/py/alchemist-nrel.svg)](https://pypi.org/project/alchemist-nrel/)
[![Tests](https://github.com/NatLabRockies/ALchemist/actions/workflows/tests.yml/badge.svg)](https://github.com/NatLabRockies/ALchemist/actions/workflows/tests.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

ALchemist is a modular Python toolkit that brings active learning and Bayesian optimization to experimental design in chemical and materials research. It is designed for scientists and engineers who want to efficiently explore or optimize high-dimensional variable spaces—using intuitive graphical interfaces, programmatic APIs, or autonomous optimization workflows.

**NLR Software Record:** SWR-25-102

---

<img src="https://raw.githubusercontent.com/NatLabRockies/ALchemist/main/docs/images/alchemist-web.png" alt="The ALchemist web app on a simulated reactor session: the variables and 16 results on the left, a contour plot of the model's predicted yield over temperature and pressure in the centre, and the Gaussian process model settings and cross-validation metrics on the right." width="100%">

## Documentation

Full user guide and documentation:  
[https://natlabrockies.github.io/ALchemist/](https://natlabrockies.github.io/ALchemist/)

---

## Overview

**Key Features:**

- **Flexible variable space definition**: Real, integer, categorical, and discrete variables with bounds or allowed value sets
- **Full DoE suite**: Space-filling (LHS, Sobol), classical RSM (CCD, Box-Behnken, Full/Fractional Factorial), screening (Plackett-Burman, GSD), and optimal designs (D/A/I-optimal with five exchange algorithms)
- **AI-assisted experimental design**: LLM-powered effect selection for optimal designs — integrates with OpenAI, local Ollama models, and Edison Scientific literature search
- **Probabilistic surrogate modeling**: Gaussian process regression via BoTorch (Matern, RBF, IBNN kernels) or scikit-learn backends
- **Advanced acquisition strategies**: Expected Improvement (including the log form), Probability of Improvement and Upper Confidence Bound, batch qEI and qUCB, and qNegIntegratedPosteriorVariance for exploration
- **Multi-objective optimization**: Pareto frontier visualization and hypervolume convergence tracking
- **Modern web interface**: React-based UI with FastAPI backend for seamless active learning workflows
- **Desktop GUI (deprecated)**: CustomTkinter desktop application for offline optimization, installed with the `[desktop]` extra
- **Session management**: Save/load optimization sessions with audit logs for reproducibility
- **Multiple interfaces**: No-code GUI, Python Session API, or REST API for different use cases
- **Autonomous optimization**: Staged experiments API enables human-out-of-the-loop operation for reactor-in-the-loop and automated laboratory workflows
- **Experiment tracking**: CSV logging, reproducible random seeds, and comprehensive audit trails  

**Architecture:**

ALchemist is built on a clean, modular architecture:

- **Core Session API**: Headless Bayesian optimization engine (`alchemist_core`) that powers all interfaces
- **Desktop Application (deprecated)**: CustomTkinter GUI using the Core Session API, designed for human-in-the-loop and offline optimization
- **REST API**: FastAPI server providing a thin wrapper around the Core Session API for remote access; includes a staged experiments queue for autonomous reactor-in-the-loop workflows
- **Web Application**: React UI consuming the REST API, supporting both interactive and autonomous optimization workflows; real-time WebSocket event streaming for live session monitoring

Session files (JSON format) are fully interoperable between desktop and web applications, enabling seamless workflow transitions.

---

## Use Cases

- **Interactive Optimization**: Use desktop or web GUI for manual experiment design and human-in-the-loop optimization
- **Programmatic Workflows**: Import the Session API in Python scripts or Jupyter notebooks for batch processing
- **Autonomous Optimization**: Use the REST API to integrate ALchemist with automated laboratory equipment for real-time process control
- **Remote Monitoring**: Web dashboard provides read-only monitoring mode when ALchemist is being remote-controlled

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

We appreciate your feedback and bug reports to help improve ALchemist!

---

## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

---

## Repository

[https://github.com/NatLabRockies/ALchemist](https://github.com/NatLabRockies/ALchemist)

