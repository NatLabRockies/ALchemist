# Optimal design and AI effect suggestion

This page generates a design that is optimal for a model you name, and shows how to ask an AI model which effects that model should contain.

![The Optimal Design (OED) panel in Quick mode with a Quadratic model, a 2×p run count, the D-optimal criterion and the Fedorov algorithm. Below the Generate Optimal Design button, a Design Quality box reports D-efficiency 47.9%, A-efficiency 26.1%, Criterion D and 30 runs, above a table of the 30 generated points.](../../images/guide/optimal-design.png)

*An optimal design of 30 runs for a quadratic model on four simulated reactor variables. The design was not seeded, so a new click gives slightly different efficiencies.*

## When to use it

A space-filling design ([Initial design](../initial-design.md)) assumes nothing about the response. An optimal design assumes a model, such as temperature, pressure, their interaction and temperature squared, and picks the runs that estimate those terms best. Use it when experiments are expensive, when you can name the terms you expect to matter, or when constraints rule out the corners of a classical design. The criteria and algorithms are in [Design of experiments](../../concepts/design-of-experiments.md#optimal-designs).

## 1. Generate an optimal design

The **Optimal Design (OED)** panel is in the left column, below **Initial Design (DoE)**. It needs at least one variable.

1. Under **Model Specification**, choose **Quick** or **Custom Effects**.
2. In **Quick**, pick a model from the menu. In **Custom Effects**, tick the terms you want, grouped as Main Effects, Interactions and Quadratic (each group has all and none links), or type more in "Additional effects (comma-separated)", such as `Temperature*Pressure` or `Temperature**2`.
3. Choose **Preview Model Terms** to see the number of model columns p, the minimum number of runs (p), the recommended number (2p) and the list of terms. It also sets the **Fixed count** to the recommended number.
4. Under **Run Count**, use **Multiplier (×p)** (a slider from 1 to 5) or **Fixed count**.
5. Pick the **Optimality Criterion** and the **Algorithm**. Under **Advanced** you can change the grid levels, the maximum iterations and the random seed.
6. Choose **Generate Optimal Design**. The **Design Quality** box shows the D-efficiency, the A-efficiency, the criterion and the number of runs, and the points appear in a table.
7. Choose **Stage** to queue the points, with the reason "Optimal DoE (D-optimal, fedorov)", for **Add Point...**; or **CSV** to save `optimal_design_<criterion>_<N>pts.csv`. Saving works as for the [initial design](../initial-design.md#1-generate-a-design).

Nothing is added to the session's data until you record results. Generating a design does add an entry to the audit log.

| **Quick** menu | Terms in the model |
|---|---|
| **Linear (main effects only)** | Intercept and one term per variable. |
| **Interaction (main + pairwise)** | Linear, plus every pair of variables. |
| **Quadratic (main + pairwise + squared)** | Interaction, plus a squared term for each real and integer variable. This is the default. |

A categorical variable adds one indicator column for each category after the first, so p counts columns, not variables.

| **Optimality Criterion** | `criterion` | What it favors |
|---|---|---|
| **D-optimal (parameter estimation)** | `D` | A small confidence region for the model's coefficients. The default. |
| **A-optimal (min avg variance)** | `A` | A small average variance of the coefficients. |
| **I-optimal (min prediction variance)** | `I` | A small average prediction variance over the space. |

| **Algorithm** | `algorithm` |
|---|---|
| **Fedorov (recommended)** | `fedorov`, the default |
| **Modified Fedorov** | `modified_fedorov` |
| **DetMax (best quality)** | `detmax` |
| **Simple Exchange** | `simple_exchange` |
| **Sequential (fastest)** | `sequential` |

!!! warning "Known issue"
    T10: **DetMax (best quality)** and **Modified Fedorov** run the same exchange as **Fedorov (recommended)**, so the label "best quality" does not mean a better design. What each algorithm does, and where the code differs, is on the Concepts page: [Optimal designs](../../concepts/design-of-experiments.md#optimal-designs).

=== "Python"

    ```python
    from alchemist_core import OptimizationSession

    session = OptimizationSession()
    session.add_variable("temperature", "real", bounds=(200.0, 300.0), unit="°C")
    session.add_variable("pressure", "real", bounds=(20.0, 80.0), unit="bar")
    session.add_variable("h2_co2", "real", bounds=(2.0, 4.0))

    info = session.get_optimal_design_info(model_type="quadratic")
    print(info["p_columns"], info["n_points_minimum"], info["n_points_recommended"])

    points, design = session.generate_optimal_design(
        model_type="quadratic", p_multiplier=2.0, criterion="D", algorithm="fedorov", random_seed=7
    )
    print(len(points), round(design["D_eff"], 1), round(design["A_eff"], 1))
    print(points[0])
    ```

    To name the effects yourself, pass `effects` instead of `model_type`, and `n_points` instead of `p_multiplier`:

    ```python
    points, design = session.generate_optimal_design(
        effects=["temperature", "pressure", "h2_co2", "temperature*pressure", "temperature**2"],
        n_points=12,
        criterion="I",
        algorithm="sequential",
        random_seed=7,
    )
    print(design["model_terms"])
    ```

    Each point is a dictionary of variable values. Run it, then record the result with `session.add_experiment(point, output=...)`.

=== "REST"

    ```http
    POST /api/v1/sessions/{session_id}/optimal-design/info
    Content-Type: application/json

    {"model_type": "quadratic"}
    ```

    The response holds `model_terms`, `p_columns`, `n_points_minimum` and `n_points_recommended`.

    ```http
    POST /api/v1/sessions/{session_id}/optimal-design
    Content-Type: application/json

    {"model_type": "quadratic", "p_multiplier": 2.0, "criterion": "D", "algorithm": "fedorov", "random_seed": 7}
    ```

    The response holds `points`, `n_points`, `design_info` (`criterion`, `algorithm`, `score`, `D_eff`, `A_eff`, `p_columns`, `n_runs`, `model_terms`) and `feasibility`, which is `null` unless the session has input constraints. To queue the points as **Stage** does, post them to the staged-experiments route shown on the [initial design](../initial-design.md#from-python-or-rest) page.

## Defaults and limits

Send either `model_type` or `effects`, and either `n_points` or `p_multiplier`. Python and REST have no default for either pair.

| Setting | Web app | Python and REST |
|---|---|---|
| Model | Quick, Quadratic | none; `model_type` is `linear`, `interaction` or `quadratic` |
| Run count | **Multiplier (×p)** at 2; **Fixed count** starts at 12 and takes 1 to 10,000 | `p_multiplier` from 1.0 to 10.0 (the slider stops at 5), or `n_points` from 1 to 10,000; `ceil(p_multiplier * p)` runs |
| Criterion | D-optimal | `criterion="D"` |
| Algorithm | Fedorov | `algorithm="fedorov"` |
| Grid levels | 5, from 2 to 20 | `n_levels=5` |
| Max iterations | 200, from 10 to 10,000 | `max_iter=200` |
| Random seed | blank | `random_seed=None`, a new design every call |

| Situation | What happens |
|---|---|
| Fewer runs than model columns | The call fails: "n_points=5 is fewer than the number of model columns p=10". Use at least p runs; 2p is the recommendation. |
| More runs than candidate points | The call fails and asks for a finer grid. Raise the grid levels. |
| A variable appears in no effect | The exchange cannot see it. Its values are spread evenly over its range, in shuffled order, so the variable varies but is not optimized. |
| A categorical variable | Allowed, and the code marks it experimental. For screening a categorical variable, **Full Factorial** or **GSD** in the initial design are better established. |
| A squared term on a discrete variable with two values | Refused. In coded units the square equals the intercept, so the design matrix is singular. |
| Input constraints | The candidates are cut to the feasible region and extended with points on its boundary; see [Constraints](../constraints.md). |

!!! info "How ALchemist does it"
    `run_optimal_design` in `alchemist_core/utils/optimal_design.py` picks the runs from a candidate grid with `n_levels` points per continuous variable. It starts from a greedy maximin-distance selection and swaps points while the criterion improves, up to `max_iter` passes. pyDOE computes the D- and A-efficiencies, in percent, with factors coded to -1 and +1, and both are reported whichever criterion you chose.

## 2. Ask an AI model which effects to include

In **Custom Effects** mode the panel has a collapsed section, **Suggest effects with AI**, marked Beta. It sends your variable names and types, the list of effects it may choose from, and a description you write to a language model, which picks effects from that list and says why. No experimental results are sent. Treat the answer as a starting point; you decide what the chemistry supports.

Install the extra first. It adds `openai`, `httpx` and `edison-client`.

```bash
pip install "alchemist-nrel[llm]"
```

1. Choose **Custom Effects**, then open **Suggest effects with AI**.
2. Pick a **Provider**: **OpenAI** or **Ollama (local)**.
3. For **OpenAI**, set the **Model** (it starts at `gpt-4o`) and type an **API Key**, or leave the key blank to use a saved key or the `OPENAI_API_KEY` environment variable of the server. For **Ollama (local)**, check the **Base URL** (it starts at `http://localhost:11434/v1`), choose **Refresh**, and pick a **Model** from the list.
4. Optionally tick **Edison Scientific**, a literature search that grounds the suggestion in cited papers. It takes about 10 to 20 minutes. Give an **Edison API Key** (or set `EDISON_API_KEY` on the server) and a **Search mode**: **Literature (standard)**, **Literature (high accuracy)** or **Precedent**.
5. Describe the system in **Experiment description**, for example "Methanol synthesis over a Cu/ZnO catalyst; maximize yield at 200 to 300 °C".
6. Choose **Suggest effects**. Progress messages appear while it runs, and an Edison search shows a link to follow its progress. Cancel stops the wait.
7. The result lists the suggested effects as chips. A chip marked with a warning sign names an effect that is not in your variable set. Choose Apply to checkboxes to replace the ticked terms with the suggestion. Reasoning, Literature context (Edison) and Sources open below it.

Without Edison, any sources in the answer come from the model's memory, and the panel marks them as not retrieved from a literature database: check them before you cite them. The server keeps an Edison result in memory, outside the session file, so asking again with the same description is instant; the link "Re-run Edison" asks for a fresh search. Edison gives up after 20 minutes by default, and the suggestion then continues without literature.

**Save provider settings** writes the Ollama URL and any key you typed to `~/.alchemist/config.json` on the machine that runs the ALchemist server, readable by its owner only. It writes only what is in the panel at that moment, so type your key again before you save, or a key saved earlier is dropped.

=== "Python"

    There is no `OptimizationSession` method for this step. The suggestion is a REST route that streams its progress, so call it on a running ALchemist server and then pass the effects to `generate_optimal_design`. This block calls a provider, so it is not run with the docs tests.

    ```{ .python .no-run }
    import json
    import httpx

    body = {
        "structuring_provider": {"provider": "openai", "model": "gpt-4o"},
        "system_context": "Methanol synthesis over a Cu/ZnO catalyst; maximize yield.",
    }
    url = f"http://localhost:8000/api/v1/llm/suggest-effects/{session_id}"
    with httpx.stream("POST", url, json=body, timeout=None) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                event = json.loads(line[len("data: "):])
                if event["status"] == "complete":
                    effects = event["result"]["effects"]
    points, design = session.generate_optimal_design(effects=effects, p_multiplier=2.0)
    ```

=== "REST"

    ```http
    POST /api/v1/llm/suggest-effects/{session_id}
    Content-Type: application/json

    {
      "structuring_provider": {"provider": "openai", "model": "gpt-4o"},
      "system_context": "Methanol synthesis over a Cu/ZnO catalyst; maximize yield.",
      "edison_config": {"job_type": "literature"}
    }
    ```

    Leave out `edison_config` to skip the literature search; it also takes `api_key`, `timeout_secs` (60 to 3600, default 1200) and `force_refresh`. A provider's `api_key` is optional when a key is saved or in the server's environment; `base_url` applies to `ollama`. The answer is a stream of server-sent events, each `data: ` followed by JSON with a `status`.

    | `status` | Meaning |
    |---|---|
    | `searching_literature`, `searching_literature_progress` | The Edison search is running; `trajectory_url` links to it. |
    | `literature_complete`, `literature_warning`, `literature_error` | The search finished, found no confident answer, or failed. The suggestion continues without literature after the last two. |
    | `structuring` | The language model is choosing effects. |
    | `complete` | `result` holds `effects`, `reasoning`, `confidence`, `sources`, `disclaimer` and `literature_context`. |
    | `error` | `message` says what failed, for example that the `[llm]` extra is not installed. |

    `GET /api/v1/llm/config` returns the saved settings with keys masked, and `GET /api/v1/llm/ollama/models` lists the models a running Ollama has.

!!! info "How ALchemist does it"
    `api/services/llm_service.py` fixes the instructions: choose only from the listed effects, pick the smallest set it can justify, and cite only what the literature context contains. OpenAI is asked for a strict JSON schema through the Responses API with `store=False`, and Ollama for JSON mode through its OpenAI-compatible chat endpoint. Effects outside your variable set are dropped when you apply the suggestion.
