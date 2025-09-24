"""
Simulator validation integration utilities.

This package modularizes the original validate_simulator_functional.py script
into focused modules for fetching data, hyperparameters, comparisons, and
diagnostic/report generation. It is intended for integration-style testing
of the Yuma simulator against real metagraph data.
"""

from .runner import validate_simulator, print_validation_results
from .hyperparams import (
    get_top_subnets_by_tao_emission,
    fetch_multiple_subnets_hyperparameters,
)

__all__ = [
    "validate_simulator",
    "print_validation_results",
    "get_top_subnets_by_tao_emission",
    "fetch_multiple_subnets_hyperparameters",
]

