# Models

Gaussian Process regression models for Bayesian optimization. ALchemist supports two backends: BoTorch (PyTorch-based) and scikit-learn.

---

## BoTorch Model

PyTorch-based Gaussian Process implementation with GPU support and advanced features.

::: alchemist_core.models.botorch_model.BoTorchModel
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - train
        - predict
        - get_hyperparameters

---

## Sklearn Model

Scikit-learn based Gaussian Process implementation for CPU-only workflows.

::: alchemist_core.models.sklearn_model.SklearnModel
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - train
        - predict
        - get_hyperparameters

---

## See Also

- **[OptimizationSession](session_class.md)** - High-level model training interface
- **[BoTorch Guide](../modeling/botorch.md)** - Detailed BoTorch configuration
- **[Sklearn Guide](../modeling/skopt.md)** - Detailed sklearn configuration
