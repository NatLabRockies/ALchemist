#![ALchemist](assets/NEW_LOGO_LIGHT.png){ width="400"}

**ALchemist: Active Learning Toolkit for Chemical and Materials Research**

ALchemist is a modular Python toolkit that brings active learning and Bayesian optimization to experimental design in chemical and materials research. It is designed for scientists and engineers who want to efficiently explore or optimize high-dimensional variable spaces—using intuitive graphical interfaces, programmatic APIs, or autonomous optimization workflows.

**NLR Software Record:** SWR-25-102

---

## Key Features

- **Flexible variable space definition:** Real, integer, categorical, and discrete variables with bounds or allowed value sets.

- **Full DoE suite:** Space-filling (LHS, Sobol), classical RSM (CCD, Box-Behnken, Full/Fractional Factorial), screening (Plackett-Burman, GSD), and optimal designs (D/A/I-optimal with five exchange algorithms).

- **AI-assisted experimental design:** LLM-powered effect selection for optimal designs — integrates with OpenAI, local Ollama models, and Edison Scientific literature search.

- **Probabilistic surrogate modeling:** Gaussian process regression via BoTorch (Matern, RBF, IBNN kernels) or scikit-learn backends.

- **Advanced acquisition strategies:** Efficient sampling using qEI, qPI, qUCB, and qNegIntegratedPosteriorVariance.

- **Multi-objective optimization:** Pareto frontier visualization and hypervolume convergence tracking.

- **Modern web interface:** React-based UI with FastAPI backend for seamless active learning workflows.

- **Desktop GUI:** CustomTkinter desktop application for offline optimization.

- **Session management:** Save/load optimization sessions with audit logs for reproducibility.

- **Multiple interfaces:** No-code GUI, Python Session API, or REST API for different use cases.

- **Autonomous optimization:** Staged experiments API enables human-out-of-the-loop operation for reactor-in-the-loop and automated laboratory workflows.

- **Experiment tracking:** CSV logging, reproducible random seeds, and comprehensive audit trails.

---

## Architecture

ALchemist is built on a clean, modular architecture:

- **Core Session API**: Headless Bayesian optimization engine (`alchemist_core`) that powers all interfaces

- **Desktop Application**: CustomTkinter GUI using the Core Session API, designed for human-in-the-loop and offline optimization

- **REST API**: FastAPI server providing a thin wrapper around the Core Session API for remote access

- **Web Application**: React UI consuming the REST API, supporting both interactive and autonomous optimization workflows

**Session Compatibility**: Optimization sessions are fully interoperable between desktop and web applications. Session files (JSON format) can be created, edited, and loaded in either interface, enabling seamless workflow transitions.

**Use Cases**:

- **Interactive Optimization**: Use desktop or web GUI for manual experiment design and human-in-the-loop optimization

- **Programmatic Workflows**: Import the Session API in Python scripts or Jupyter notebooks for batch processing

- **Autonomous Optimization**: Use the REST API to integrate ALchemist with automated laboratory equipment for real-time process control

- **Remote Monitoring**: Web dashboard provides read-only monitoring mode when ALchemist is being remote-controlled

---

## Installation

We recommend using [Anaconda](https://www.anaconda.com/products/distribution) to manage your Python environments.

**Requirements:** Python 3.11 or higher

**1. Create a new environment:**
```bash
conda create -n alchemist-env python=3.11
conda activate alchemist-env
```

**2. Install ALchemist:**

**From PyPI (recommended):**
```bash
pip install alchemist-nrel
```

This installs the latest stable release with pre-built web application files. Both desktop and web applications are ready to use immediately.

### Running ALchemist

**Web Application:**
```bash
alchemist-web
```
Open your browser to http://localhost:8000

**Desktop Application:**
```bash
alchemist
```
The desktop GUI launches directly.

---

## Advanced Installation

### Latest Development Version

**Desktop app only (GitHub install):**
```bash
pip install git+https://github.com/NatLabRockies/ALchemist.git
```

This installs the latest unreleased version. 
> *Note: The web application is not available with this method because static build files are not included in the repository.*

**Web and desktop apps (clone and build):**
```bash
git clone https://github.com/NatLabRockies/ALchemist.git
cd ALchemist
pip install -e .
cd alchemist-web
npm install
npm run build
```

This builds the web application from source, giving you the latest unreleased version of both apps.

### Docker Deployment

ALchemist can be deployed as a Docker container for production environments.

**Prerequisites**: [Docker Desktop](https://docs.docker.com/desktop/) installed and running

**Quick start:**
```bash
git clone https://github.com/NatLabRockies/ALchemist.git
cd ALchemist/docker
docker-compose up -d
```

**Access the application:**

- Web UI: http://localhost:8000/app

- API Docs: http://localhost:8000/docs

- Health Check: http://localhost:8000/health

**Stop the container:**
```bash
docker-compose down
```

---

Use the sidebar to navigate through the documentation. See [Getting Started](setup/variable_space.md) to define your variable space and generate initial experiments.