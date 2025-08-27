# Simulator Validation (Integration)

This package modularizes the original `validate_simulator_functional.py` script into
focused modules. It is intended for integration-level validation of the Yuma simulator
against real metagraph data.

- `data.py`: Fetches and prepares metagraph data
- `hyperparams.py`: Fetches and normalizes subnet hyperparameters
- `compare.py`: Implements bonds/dividends/incentives comparisons
- `diagnostics.py`: Creates detailed failure diagnostics and reports
- `runner.py`: Orchestrates a full validation run

You can still invoke the original CLI via the root-level `validate_simulator_functional.py`.
A timestamped backup of the pre-modularized script is stored under `backups/`.
