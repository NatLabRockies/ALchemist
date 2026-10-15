# Install

This page installs ALchemist, starts the web app, and lists the optional extras, a source install and Docker.

## Requirements

ALchemist needs Python 3.11 or later; the test suite runs on 3.11, 3.12 and 3.13, on Linux, macOS and Windows. It installs PyTorch, BoTorch and scikit-learn, so the first install is a large download, most of it PyTorch. Install it into its own environment:

```bash
python -m venv alchemist-env
source alchemist-env/bin/activate      # Windows: alchemist-env\Scripts\activate
```

or, with conda:

```bash
conda create -n alchemist-env python=3.12
conda activate alchemist-env
```

## Install from PyPI

```bash
pip install alchemist-nrel
```

This installs the Python library (`alchemist_core`), the REST API and the web app, already built. Check it with:

```bash
python -c "import alchemist_core; print(alchemist_core.__version__)"
```

## Optional extras

| Extra | Install | Adds |
|---|---|---|
| `llm` | `pip install "alchemist-nrel[llm]"` | The clients for AI effect suggestion in optimal designs: OpenAI, a local Ollama model, and Edison Scientific literature search. See [Optimal design](../guide/more/optimal-design.md#2-ask-an-ai-model-which-effects-to-include). |
| `desktop` | `pip install "alchemist-nrel[desktop]"` | The desktop app, `alchemist`. It is deprecated; see [Desktop app (legacy)](../reference/desktop.md). |
| `docs` | `pip install "alchemist-nrel[docs]"` | The tools that build this site (Zensical and mkdocstrings), for working on the documentation. |
| `test` | `pip install "alchemist-nrel[test]"` | pytest and the packages the test suite needs. |

Quote the name with its brackets, as shown, so that shells such as zsh do not read the brackets as a pattern.

## Start the web app

```bash
alchemist-web
```

It prints "Access the web UI at: http://localhost:8000". Open that address in a browser. Stop the server with Ctrl+C.

| Address | Shows |
|---|---|
| `http://localhost:8000` | The web app. |
| `http://localhost:8000/api` | The API's name, version and the paths below, as JSON. |
| `http://localhost:8000/api/docs` | The interactive REST reference (Swagger). The same routes are listed in the [REST API reference](../reference/rest-api.md). |
| `http://localhost:8000/health` | `{"status": "healthy", ...}` while the server runs. |

`alchemist-web` always uses port 8000 (W37). If another program holds that port it stops with "address already in use"; see [Troubleshooting](troubleshooting.md#installing-and-starting) for the fix and for running on another port.

The server keeps its sessions in `cache/sessions`, and the 30-second recovery copies of open sessions in `cache/recovery`, under the folder you start it from. Start it from the same folder each time so that **Join** finds the same sessions, and save the sessions you want to keep to a file; see [Sessions and audit log](../guide/more/sessions.md).

`alchemist-web` listens on every network interface of the computer, and it has no login. Anyone who can reach your computer on port 8000 can open the app and its sessions. On a shared network, block the port in your firewall, or start the server on the local interface only:

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

## Use it from Python only

Nothing has to run in the background to use the library:

```python
from alchemist_core import OptimizationSession

session = OptimizationSession()
session.add_variable("temperature", "real", bounds=(200.0, 300.0), unit="°C")
print(session.get_search_space_summary()["n_variables"])
```

The [Python quickstart](../python/quickstart.md) runs a whole campaign this way.

## Install from source

A clone gives you the unreleased version. The repository does not hold the built web app, so build it once with Node.js 20.19 or later (or 22.12 or later):

```bash
git clone https://github.com/NatLabRockies/ALchemist.git
cd ALchemist
pip install -e ".[test]"
cd alchemist-web
npm ci
npm run build
cd ..
alchemist-web
```

The server serves the build from `alchemist-web/dist`. Without it, `http://localhost:8000` returns the API's JSON instead of the app. The same is true of `pip install git+https://github.com/NatLabRockies/ALchemist.git`, which installs the library and the API but no web app. `alchemist-web --dev` restarts the server when a Python file changes.

## Docker

The `docker/` folder of the repository builds an image with the API and the web app:

```bash
git clone https://github.com/NatLabRockies/ALchemist.git
cd ALchemist/docker
docker compose up --build
```

Open `http://localhost:8000` as before. The compose file keeps the server's `cache/` and `logs/` folders in the repository root, so sessions survive a restart of the container, and checks `/health` every 30 seconds. `docker compose up -d` runs it in the background and `docker compose down` stops it. To publish it on another host port, change the left side of `"8000:8000"` in `docker/docker-compose.yml`.

## Next

Run your first optimization on the simulated reactor: [Your first loop](first-loop.md).
