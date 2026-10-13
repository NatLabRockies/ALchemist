"""
ALchemist Core - Headless Bayesian Optimization Library

This package provides the core functionality for active learning and Bayesian
optimization workflows without UI dependencies.

Main Components:
    - OptimizationSession: High-level API for optimization workflows
    - SearchSpace: Define variable search spaces
    - ExperimentManager: Manage experimental data
    - Models: Surrogate modeling backends (sklearn, BoTorch)
    - Acquisition: Acquisition function strategies
    - AuditLog: Reproducible audit trail for optimization decisions

Example:
    ```python
    from alchemist_core import OptimizationSession

    session = OptimizationSession()
    session.add_variable("temperature", "real", bounds=(300, 500))
    session.load_data("experiments.csv", target_columns="yield")
    session.train_model(backend="sklearn")
    next_point = session.suggest_next(strategy="EI", goal="maximize")
    ```

Version: 0.3.4
"""

__version__ = "0.3.4"
__author__ = "Caleb Coatney"
__email__ = "caleb.coatney@nlr.gov"

# Core data structures
from alchemist_core.data.search_space import SearchSpace
from alchemist_core.data.experiment_manager import ExperimentManager

# Audit log and session metadata
from alchemist_core.audit_log import AuditLog, SessionMetadata, AuditEntry

# Event system
from alchemist_core.events import EventEmitter

# Configuration and logging
from alchemist_core.config import configure_logging, get_logger, set_verbosity

# High-level session API
from alchemist_core.session import OptimizationSession

# Public API
__all__ = [
    # Session API
    "OptimizationSession",
    # Data structures
    "SearchSpace",
    "ExperimentManager",
    # Audit log
    "AuditLog",
    "SessionMetadata", 
    "AuditEntry",
    # Event system
    "EventEmitter",
    # Configuration
    "configure_logging",
    "get_logger",
    "set_verbosity"
]
