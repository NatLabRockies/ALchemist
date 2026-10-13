# Getting Started

ALchemist provides both a **web application** (browser-based) and **desktop application** (native GUI). Both interfaces offer the same functionality with nearly identical layouts and workflows. Choose whichever fits your needs - sessions are fully interoperable between both.

---

## Launching ALchemist

### Web Application

```bash
alchemist-web
```

Open your browser to `http://localhost:8000`

### Desktop Application

```bash
alchemist
```

The desktop GUI launches directly.

---

## Basic Workflow

All Bayesian optimization projects in ALchemist follow the same basic steps:

1. **Create or load a session** - Your optimization project container
2. **Define variables** - Set up your search space (see [Variable Space](../setup/variable_space.md))
3. **Add initial data** - Import experiments or generate space-filling designs
4. **Train a model** - Fit Gaussian Process to your data (see [Modeling](../modeling/botorch.md))
5. **View diagnostics** - Check model quality (see [Visualizations](../visualizations/parity_plot.md))
6. **Get suggestions** - Run acquisition functions for next experiments (see [Acquisition](../acquisition/botorch.md))
7. **Repeat** - Add new data and iterate steps 4-6

---

## Key Features

**Session Management**: Create, save, and load optimization projects. Sessions include your variable definitions, experimental data, trained models, and full audit logs.

**Flexible Variable Types**: Define continuous (bounded ranges), discrete (specific values), or categorical (named options) variables.

**Data Import/Export**: Add experiments manually or import from CSV. Export suggestions for lab execution.

**Model Training**: Choose between BoTorch (advanced, GPU-ready) or scikit-learn (lightweight) backends with automatic hyperparameter optimization.

**Visualizations**: Parity plots, Q-Q plots, calibration curves, and metrics evolution help you understand model performance.

**Acquisition Strategies**: Expected Improvement, Upper Confidence Bound, Probability of Improvement, and Thompson Sampling.

---

## What's Different Between Web and Desktop?

Very little - they share the same core functionality:

| Feature | Web App | Desktop App |
|---------|---------|-------------|
| User Interface | Browser-based | Native GUI |
| Functionality | Identical | Identical |
| Sessions | Fully compatible | Fully compatible |
| Offline Use | Requires backend server | Works fully offline |

Choose the **web app** for remote access or team collaboration. Choose the **desktop app** for offline work or if you prefer native applications.

---

## Where to Go Next

This guide covers launching ALchemist and the basic workflow. For detailed information on each step:

- **[Variable Space Setup](../setup/variable_space.md)** - Define your optimization problem

- **[Model Training](../modeling/botorch.md)** - Configure and train Gaussian Processes

- **[Visualizations](../visualizations/parity_plot.md)** - Interpret model diagnostics

- **[Acquisition Functions](../acquisition/botorch.md)** - Choose how to explore your space

- **[Session Management](../reproducibility/sessions.md)** - Save, load, and share your work

- **[REST API](../api/rest.md)** - Programmatic access for automation


