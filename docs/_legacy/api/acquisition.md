# Acquisition Functions

Acquisition functions guide the selection of next experiments in Bayesian optimization. They balance exploration (reducing uncertainty) and exploitation (targeting optimal regions).

---

## BoTorch Acquisition

PyTorch-based acquisition functions with batch support.

::: alchemist_core.acquisition.botorch_acquisition.BoTorchAcquisition
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - select_next
        - find_optimum

---

## Skopt Acquisition

Scikit-optimize based acquisition functions.

::: alchemist_core.acquisition.skopt_acquisition.SkoptAcquisition
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - select_next
        - find_optimum

---

## Available Strategies

### Single-Point Strategies

| Strategy | Code | Description | Best For |
|----------|------|-------------|----------|
| **Expected Improvement** | `'EI'` | Balances exploration/exploitation | General use |
| **Log EI** | `'LogEI'` | Numerically stable EI | Noisy data |
| **Probability of Improvement** | `'PI'` | Conservative improvement | Risk-averse |
| **Log PI** | `'LogPI'` | Numerically stable PI | Noisy data |
| **Upper Confidence Bound** | `'UCB'` | More exploratory | Unknown spaces |

### Batch Strategies

| Strategy | Code | Description | Best For |
|----------|------|-------------|----------|
| **Batch Expected Improvement** | `'qEI'` | Parallel experiments | Lab workflows |
| **Batch UCB** | `'qUCB'` | Parallel + exploratory | Exploration |
| **Negative Integrated Posterior Variance** | `'qNIPV'` | Pure exploration | Model improvement |

---

## See Also

- **[OptimizationSession](session_class.md)** - High-level acquisition interface
- **[BoTorch Acquisition Guide](../acquisition/botorch.md)** - Detailed strategy explanations
- **[Models](models.md)** - Model training for acquisition
