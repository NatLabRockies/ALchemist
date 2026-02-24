# AI-Assisted Effect Selection

ALchemist can leverage large language models (LLMs) to suggest which model terms — main effects, interactions, and quadratic terms — are likely important for your experimental system. This is especially useful when setting up an [Optimal Design](doe_optimal.md) and you are unsure which effects to include.

---

## Overview

The AI effect suggestion workflow:

1. **(Optional)** Query **Edison Scientific** for cited scientific literature relevant to your system.
2. Pass that literature context (or just your own system description) to an LLM — **OpenAI** (e.g., `gpt-4o`) or a locally-running **Ollama** model.
3. The LLM selects from your specific variable list (main effects, interactions, quadratics) and returns a structured recommendation with reasoning and confidence levels.
4. You review and accept or modify the suggested effects before generating the design.

> **Important:** AI suggestions are a starting point — always review the reasoning and apply your domain expertise before accepting. The model is constrained to choose only from the effects that make sense for your variable space.

---

## Installation

LLM features require additional dependencies. Install with:

```bash
pip install 'alchemist-nrel[llm]'
```

This adds `openai`, `httpx`, and `edison-client` to your environment. These are not installed by default.

---

## Provider Setup

### OpenAI

1. Obtain an API key from [platform.openai.com](https://platform.openai.com/api-keys).
2. In the web UI, open the AI Suggest panel, select **OpenAI**, enter your API key, and choose a model.
3. Your key is saved to `~/.alchemist/config.json` for future sessions.

**Supported models:** Any OpenAI model that supports structured outputs, e.g.:
- `gpt-4o` (recommended — best reasoning)
- `gpt-4-turbo`
- `gpt-4.1`

### Ollama (local models)

[Ollama](https://ollama.com/) lets you run open-source models entirely on your machine — no API key needed.

1. Install Ollama and pull a model:
   ```bash
   ollama pull llama3.2
   ```
2. In the web UI, select **Ollama** and click **Fetch Models** to auto-discover available models.
3. Use a model with at least 8B parameters for reliable effect selection. Larger models (e.g., 70B) give more accurate reasoning.

**Default base URL:** `http://localhost:11434` — override this if Ollama runs on a different host.

### Edison Scientific (literature search)

[Edison Scientific](https://edisonsci.ai/) is an AI-powered scientific literature search platform built on PaperQA3. When enabled, ALchemist submits a query about your experimental system before calling the structuring LLM. The LLM then grounds its recommendations in real citations from the literature.

1. Obtain an Edison API key from your Edison account.
2. Enter your key in the Edison section of the AI Suggest panel.
3. Choose a job type:
   - `literature` — standard PaperQA3 search (recommended; ~5–15 minutes)
   - `literature_high` — high-reasoning mode (slower, deeper analysis)
   - `precedent` — precedent-style search ("has anyone done X?")

> **Caching:** Edison results are cached per session for 15 minutes. Re-opening the suggest panel with the same query will use the cached result instantly. Use **Force Refresh** to re-run the search.

> **Timeout:** If Edison does not return within the timeout (default 20 minutes), ALchemist proceeds without literature context. The structuring LLM still runs — its suggestions just won't cite specific papers.

---

## Web UI Walkthrough

1. In the **Optimal Design** panel, click **Suggest Effects with AI**.
2. Select your provider (OpenAI or Ollama) and configure API key / model.
3. *(Optional)* Enable Edison Scientific and select a job type.
4. Enter a **System Context** describing your experimental objective:
   > *"Fischer-Tropsch synthesis over supported metal catalysts. Optimizing C5+ liquid fuel selectivity at 250 °C, 20 bar, H₂/CO = 2."*
5. Click **Suggest Effects**.
6. While streaming, you'll see status updates (Edison search, LLM call, etc.).
7. Review the results:
   - **Suggested effects** with reasoning for each
   - **Confidence levels** (high / medium / low)
   - **Literature citations** (if Edison was used)
8. Click an effect to select or deselect it.
9. Proceed to generate the optimal design with your chosen effects.

---

## REST API

The suggest-effects endpoint streams Server-Sent Events (SSE):

```http
POST /api/v1/llm/suggest-effects/{session_id}
Content-Type: application/json

{
  "structuring_provider": {
    "provider": "openai",
    "model": "gpt-4o",
    "api_key": "sk-..."
  },
  "system_context": "CO2 electroreduction on Cu catalysts, maximizing ethylene faradaic efficiency.",
  "edison_config": {
    "job_type": "literature",
    "api_key": "edi-..."
  }
}
```

**Without Edison:**
```json
{
  "structuring_provider": {
    "provider": "ollama",
    "model": "llama3.2",
    "base_url": "http://localhost:11434"
  },
  "system_context": "Pd-catalyzed Suzuki coupling. Maximizing yield at ambient temperature."
}
```

**SSE event stream** (each line is a JSON object):
```
data: {"status": "starting", "message": "Submitting Edison search..."}
data: {"status": "edison_searching", "trajectory_url": "https://..."}
data: {"status": "structuring", "message": "LLM structuring call in progress..."}
data: {"status": "complete", "result": { "effects": [...], "reasoning": [...], ... }}
```

### Config management

```http
GET /api/v1/llm/config
```
Returns saved OpenAI, Ollama, and Edison configuration.

```http
POST /api/v1/llm/config
{"openai": {"api_key": "sk-..."}, "ollama": {"base_url": "http://localhost:11434"}}
```
Saves configuration to `~/.alchemist/config.json`.

### Ollama model discovery

```http
GET /api/v1/llm/ollama/models?base_url=http://localhost:11434
```

---

## Interpreting Results

The AI response includes:

- **effects** — list of suggested effect strings (in ALchemist's effect syntax)
- **reasoning** — per-effect explanation of why it was selected
- **confidence** — high / medium / low per effect
- **sources** — citation list (only populated when Edison was used and found relevant papers)
- **disclaimer** — reminds the user that AI suggestions should be reviewed critically

**Example response excerpt:**
```json
{
  "effects": ["Temperature", "Pressure", "Temperature*Pressure", "Temperature**2"],
  "reasoning": [
    {"effect": "Temperature", "reason": "Rate-determining step is strongly temperature-activated..."},
    {"effect": "Temperature*Pressure", "reason": "Syngas H₂/CO ratio shifts with temperature × pressure interactions..."}
  ],
  "confidence": [
    {"effect": "Temperature", "level": "high"},
    {"effect": "Temperature*Pressure", "level": "medium"}
  ],
  "sources": [
    "Smith et al. (2021) Catal. Sci. Technol. — Fischer-Tropsch selectivity kinetics"
  ],
  "disclaimer": "AI suggestions are based on literature and general chemical reasoning..."
}
```

---

## Privacy & Data Handling

- Your variable names and system context description are sent to the LLM provider (OpenAI or your local Ollama instance).
- API keys are stored locally in `~/.alchemist/config.json` — they are not sent to any ALchemist server.
- When using Ollama, all data stays on your machine.
- No experimental data (results, measurements) is sent to any AI provider.
