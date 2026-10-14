# OptimizationSession

The main class for Bayesian optimization workflows. Orchestrates variable space definition, data management, model training, and acquisition function execution.

---

## Class Reference

::: alchemist_core.session.OptimizationSession
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - add_variable
        - get_search_space_summary
        - generate_initial_design
        - load_data
        - add_experiment
        - get_data_summary
        - add_staged_experiment
        - get_staged_experiments
        - clear_staged_experiments
        - move_staged_to_experiments
        - train_model
        - get_model_summary
        - suggest_next
        - predict
        - save_session
        - load_session

---

## Related Classes

### ExperimentManager

Manages experimental data storage and retrieval.

::: alchemist_core.data.experiment_manager.ExperimentManager
    options:
      heading_level: 4
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - add_experiment
        - get_data
        - get_features_and_target
        - has_noise_data
        - from_csv

---

## See Also

- **[SearchSpace](search_space.md)** - Variable space management
- **[Models](models.md)** - Gaussian Process models
- **[Acquisition](acquisition.md)** - Acquisition functions
