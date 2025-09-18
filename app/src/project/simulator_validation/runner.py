import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import torch

from project.yuma_simulation._internal.simulation_utils import _run_dynamic_simulation
from project.yuma_simulation._internal.yumas import YumaSimulationNames

from .data import prepare_metagraph_data
from .hyperparams import setup_yuma_configuration
from .compare import compare_bonds, compare_dividends, compare_incentives
from .diagnostics import create_diagnostic_artifacts

logger = logging.getLogger(__name__)


def validate_simulator(
    *,
    netuid: int = 1,
    tolerance: float = 1e-4,
    num_epochs: int = 3,
    generate_diagnostics: bool = True,
    bond_penalty_override: Optional[float] = None,
    hyperparams_data: Optional[Dict[str, Any]] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    start_block: Optional[int] = None,
    end_block: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate simulator against real metagraph data."""
    logger.info(
        f"Starting validation for netuid {netuid}"
    )

    case, tested_blocks = prepare_metagraph_data(
        netuid=netuid,
        start_date=start_date,
        end_date=end_date,
        start_block=start_block,
        end_block=end_block,
        num_epochs=num_epochs,
    )
    yuma_config, is_yuma3_on, cr_info = setup_yuma_configuration(
        netuid, bond_penalty_override, hyperparams_data=hyperparams_data
    )

    yuma_simulation_name = YumaSimulationNames().YUMA3 if is_yuma3_on else YumaSimulationNames().YUMA2
    yuma_version = "YUMA3" if is_yuma3_on else "YUMA2"
    logger.info(f"Running {yuma_version} simulation...")
    # Propagate commit–reveal period to the case so bond masking can reflect chain semantics
    # Only apply if commit_reveal_enabled is true
    try:
        commit_reveal_enabled = cr_info.get("commit_reveal_enabled", False)
        commit_reveal_period = int(cr_info.get("commit_reveal_period_epochs", 0))
        
        # Only use the period if commit-reveal is actually enabled
        if commit_reveal_enabled and commit_reveal_period > 0:
            case.commit_reveal_period_epochs = commit_reveal_period
            logger.info(
                f"Commit–reveal enabled with period (epochs): {case.commit_reveal_period_epochs}"
            )
        else:
            case.commit_reveal_period_epochs = 0
            if not commit_reveal_enabled and commit_reveal_period > 0:
                logger.info(
                    f"Commit–reveal disabled, ignoring period of {commit_reveal_period} epochs"
                )
    except Exception:
        pass
    
    # Propagate bond reset behavior to the case
    try:
        bonds_reset_enabled = cr_info.get("bonds_reset_enabled", False)
        case.bonds_reset_enabled = bool(bonds_reset_enabled)
    except Exception:
        case.bonds_reset_enabled = False

    _, _, sim_bonds, sim_incentives_per_epoch, sim_normalized_dividends = _run_dynamic_simulation(
        case=case, yuma_version=yuma_simulation_name, yuma_config=yuma_config
    )

    actual_epochs = case.num_epochs
    last_epoch_idx = actual_epochs - 1
    num_sim_epochs = len(sim_bonds)
    if num_sim_epochs == 0:
        raise ValueError("Simulator produced no bond epochs")
    
    # Compare the last simulator output with the last real epoch
    sim_comparison_idx = num_sim_epochs - 1
    
    logger.info(
        f"Comparing sim_bonds[{sim_comparison_idx}] (state after epoch {sim_comparison_idx+1}) "
        f"with real epoch {last_epoch_idx} (state at start of epoch {last_epoch_idx})"
    )

    real_bonds_last = (
        case.metas[last_epoch_idx].get("bonds", None)
        if last_epoch_idx < len(case.metas)
        else None
    )
    real_incentives_last = (
        case.incentives_epochs[last_epoch_idx]
        if len(case.incentives_epochs) > last_epoch_idx
        else None
    )
    real_dividends_last = (
        case.dividends_epochs[last_epoch_idx]
        if len(case.dividends_epochs) > last_epoch_idx
        else None
    )


    validation_results: Dict[str, Any] = {
        "blocks_tested": tested_blocks,
        "epochs_tested": last_epoch_idx,
        "comparing": f"sim_epoch_{sim_comparison_idx}_with_real_epoch_{last_epoch_idx}",
        "yuma_version": yuma_version,
        "comparisons": {},
        "summary": {
            "bonds_match": True,
            "dividends_match": True,
            "incentives_match": True,
            "total_differences": 0,
        },
    }
    epoch_results: Dict[str, Any] = {}

    bonds_result = compare_bonds(
        sim_bonds,
        real_bonds_last,
        case.validators_epochs[last_epoch_idx]
        if len(case.validators_epochs) > last_epoch_idx
        else [],
        case.metas[last_epoch_idx]["hotkeys"] if last_epoch_idx < len(case.metas) else [],
        tolerance,
        case,
        sim_comparison_idx,
    )
    epoch_results["bonds"] = bonds_result
    if not bonds_result.get("matches", False):
        validation_results["summary"]["bonds_match"] = False
        validation_results["summary"]["total_differences"] += 1

    div_result = compare_dividends(
        sim_normalized_dividends,
        real_dividends_last,
        case.validators_epochs[last_epoch_idx]
        if len(case.validators_epochs) > last_epoch_idx
        else [],
        tolerance,
    )
    epoch_results["dividends"] = div_result
    if not div_result.get("matches", False):
        validation_results["summary"]["dividends_match"] = False
        validation_results["summary"]["total_differences"] += 1

    # Build miners list (hotkeys) for the epoch
    miners_epoch: list[str] = []
    if hasattr(case, 'miners_epochs'):
        try:
            if len(case.miners_epochs) > last_epoch_idx:
                miners_epoch = case.miners_epochs[last_epoch_idx]
        except Exception:
            miners_epoch = []
    if not miners_epoch:
        # Fallback: derive from miner_indices_epochs and epoch hotkeys
        try:
            miner_uids = case.miner_indices_epochs[last_epoch_idx] if len(case.miner_indices_epochs) > last_epoch_idx else []
            epoch_hotkeys = case.metas[last_epoch_idx].get('hotkeys', []) if last_epoch_idx < len(case.metas) else []
            miners_epoch = [epoch_hotkeys[uid] if 0 <= uid < len(epoch_hotkeys) else f'UID{uid}' for uid in miner_uids]
        except Exception:
            miners_epoch = []
    # Important: incentive time-series in the simulator dict are indexed by real epoch index
    # (index 0 corresponds to epoch 0 placeholder, index k to epoch k). So compare against
    # last_epoch_idx rather than sim_comparison_idx.
    inc_result = compare_incentives(
        sim_incentives_per_epoch,
        real_incentives_last,
        miners_epoch,
        tolerance,
        last_epoch_idx,
        case=case,
        yuma_config=yuma_config,
    )
    epoch_results["incentives"] = inc_result
    if not inc_result.get("matches", False):
        validation_results["summary"]["incentives_match"] = False
        validation_results["summary"]["total_differences"] += 1

    validation_results["comparisons"][f"epoch_{last_epoch_idx}"] = epoch_results

    any_mismatch = not all(
        [
            validation_results["summary"]["bonds_match"],
            validation_results["summary"]["dividends_match"],
            validation_results["summary"]["incentives_match"],
        ]
    )
    if any_mismatch and generate_diagnostics:
        diagnostic_info = create_diagnostic_artifacts(
            netuid=netuid,
            validation_results=validation_results,
            case=case,
            sim_bonds=sim_bonds,
            sim_dividends=sim_normalized_dividends,
            sim_incentives=sim_incentives_per_epoch,
            tolerance=tolerance,
            yuma_config=yuma_config,
            sim_comparison_idx=sim_comparison_idx,
        )
        validation_results["diagnostic_artifacts"] = diagnostic_info.get(
            "artifact_path", None
        )
        logger.info(
            f"Diagnostic artifacts saved to: {diagnostic_info.get('artifact_path', 'N/A')}"
        )

    return validation_results


def print_validation_results(results: Dict[str, Any]) -> None:
    print(f"Blocks tested: {results['blocks_tested']}")
    print(f"Epochs tested: {results['epochs_tested']}")
    summary = results["summary"]
    print("SUMMARY:")
    print(f"  Bonds match:     {'✓' if summary['bonds_match'] else '✗'}")
    print(f"  Dividends match: {'✓' if summary['dividends_match'] else '✗'}")
    print(f"  Incentives match:{'✓' if summary['incentives_match'] else '✗'}")
    print(f"  Total differences: {summary['total_differences']}")
    overall_success = all(
        [summary["bonds_match"], summary["dividends_match"], summary["incentives_match"]]
    )
    if overall_success:
        print("Validation passed: Simulator matches real metagraph data.")
    else:
        print("Validation failed: Simulator differs from real metagraph data")
        if results.get("diagnostic_artifacts"):
            print(f"\nDiagnostic artifacts: {results['diagnostic_artifacts']}")
            print("Check diagnostic_report.json and summary.txt for details")
        print("\nRESULTS OVERVIEW:")
        for epoch_name, epoch_data in results["comparisons"].items():
            print(f"\n{epoch_name.upper()}:")
            for metric, data in epoch_data.items():
                if "error" in data:
                    print(f"  {metric}: ✗ ERROR - {data['error']}")
                else:
                    status = "✓" if data["matches"] else "✗"
                    extra = []
                    if data.get("missing_validators"):
                        extra.append(f"missing_validators: {data['missing_validators']}")
                    if data.get("missing_miners"):
                        extra.append(f"missing_miners: {data['missing_miners']}")
                    extra_info = (", " + ", ".join(extra)) if extra else ""
                    print(
                        f"  {metric}: {status} (max_diff: {data.get('max_diff', 0):.6f}, "
                        f"mean_diff: {data.get('mean_diff', 0):.6f}, nonzero_diffs: {data.get('nonzero_diffs', 0)}{extra_info})"
                    )
