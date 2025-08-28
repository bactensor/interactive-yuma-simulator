import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def analyze_divergent_bonds_with_weights(
    case,
    sim_bonds: List[torch.Tensor],
    epoch_idx: int,
    tolerance: float,
) -> List[Dict[str, Any]]:
    divergences: List[Dict[str, Any]] = []
    if not case.weights_epochs:
        logger.warning("No weight data available for divergence analysis")
        return divergences

    simulator_fed_weights = case.weights_epochs[0]
    real_bonds = case.metas[epoch_idx].get("bonds", None) if epoch_idx < len(case.metas) else None
    real_weights_full = None
    if len(case.metas) > 0 and "W" in case.metas[0]:
        real_weights_full = case.metas[0]["W"]
    if real_bonds is None or len(sim_bonds) == 0:
        return divergences

    validators = case.validators_epochs[epoch_idx] if len(case.validators_epochs) > epoch_idx else []
    hotkeys = case.metas[epoch_idx].get("hotkeys", []) if epoch_idx < len(case.metas) else []
    validator_positions: Dict[int, int] = {}
    for i, validator_hotkey in enumerate(validators):
        try:
            uid = hotkeys.index(validator_hotkey)
            validator_positions[i] = uid
        except ValueError:
            continue

    sim_bonds_epoch = sim_bonds[-1] if sim_bonds else None
    if sim_bonds_epoch is None:
        return divergences

    for val_idx, validator_uid in validator_positions.items():
        if val_idx >= len(simulator_fed_weights) or val_idx >= sim_bonds_epoch.shape[0]:
            continue
        validator_hotkey = validators[val_idx]
        for target_uid in range(256):
            sim_bond = float(sim_bonds_epoch[val_idx, target_uid].item())
            real_bond_raw = float(real_bonds[validator_uid, target_uid].item())
            real_bond_normalized = real_bond_raw / 65535.0
            real_bonds_col_sum = (real_bonds[:, target_uid] / 65535.0).sum().item()
            real_bond_normalized = (
                real_bond_normalized / (real_bonds_col_sum + 1e-6)
                if real_bonds_col_sum > 1e-6
                else 0
            )

            bond_diff = abs(sim_bond - real_bond_normalized)
            if bond_diff > tolerance:
                simulator_fed_weight = (
                    float(simulator_fed_weights[val_idx][target_uid])
                    if val_idx < len(simulator_fed_weights)
                    else 0.0
                )
                real_weight_from_dumper = (
                    float(real_weights_full[validator_uid, target_uid].item())
                    if real_weights_full is not None
                    else 0.0
                )
                weight_diff = abs(simulator_fed_weight - real_weight_from_dumper)
                divergences.append(
                    {
                        "validator_uid": validator_uid,
                        "validator_hotkey": validator_hotkey[:10] + "...",
                        "target_uid": target_uid,
                        "bond_comparison": {
                            "simulated": sim_bond,
                            "expected": float(real_bond_normalized),
                            "difference": float(bond_diff),
                            "raw_expected": real_bond_raw,
                        },
                        "weight_comparison": {
                            "real_from_dumper": real_weight_from_dumper,
                            "simulator_fed": simulator_fed_weight,
                            "difference": weight_diff,
                        },
                    }
                )

    divergences.sort(key=lambda x: x["bond_comparison"]["difference"], reverse=True)
    return divergences


def analyze_divergent_dividends_with_stakes(
    case,
    sim_dividends: Dict,
    epoch_idx: int,
    tolerance: float,
) -> List[Dict[str, Any]]:
    divergences: List[Dict[str, Any]] = []
    if not case.stakes_epochs:
        logger.warning("No stake data available for divergence analysis")
        return divergences

    simulator_fed_stakes = case.stakes_epochs[0]
    real_dividends = (
        case.dividends_epochs[epoch_idx]
        if len(case.dividends_epochs) > epoch_idx
        else None
    )
    real_stakes_full = None
    if len(case.metas) > 0 and "S" in case.metas[0]:
        real_stakes_full = case.metas[0]["S"]
    if real_dividends is None:
        return divergences

    validators = (
        case.validators_epochs[epoch_idx] if len(case.validators_epochs) > epoch_idx else []
    )
    hotkeys = case.metas[epoch_idx].get("hotkeys", []) if epoch_idx < len(case.metas) else []

    for i, validator_hotkey in enumerate(validators):
        if i >= len(simulator_fed_stakes) or i >= len(real_dividends):
            continue
        sim_div = 0.0
        if validator_hotkey in sim_dividends:
            div_data = sim_dividends[validator_hotkey]
            if isinstance(div_data, list) and len(div_data) > 0:
                sim_div = float(div_data[-1])
            elif isinstance(div_data, (int, float)):
                sim_div = float(div_data)
        real_div = float(real_dividends[i].item())
        div_diff = abs(sim_div - real_div)
        if div_diff > tolerance:
            validator_uid = -1
            try:
                validator_uid = hotkeys.index(validator_hotkey)
            except ValueError:
                pass
            real_stake_from_dumper = 0.0
            if (
                real_stakes_full is not None
                and 0 <= validator_uid < len(real_stakes_full)
            ):
                real_stake_from_dumper = float(real_stakes_full[validator_uid].item())
            simulator_fed_stake = float(simulator_fed_stakes[i])
            divergences.append(
                {
                    "validator_index": i,
                    "validator_uid": validator_uid,
                    "validator_hotkey": validator_hotkey[:10] + "...",
                    "dividend_comparison": {
                        "simulated": float(sim_div),
                        "expected": float(real_div),
                        "difference": float(div_diff),
                    },
                    "stake_comparison": {
                        "real_from_dumper": float(real_stake_from_dumper),
                        "simulator_fed": float(simulator_fed_stake),
                        "difference": float(
                            abs(simulator_fed_stake - real_stake_from_dumper)
                        ),
                    },
                }
            )

    divergences.sort(key=lambda x: x["dividend_comparison"]["difference"], reverse=True)
    return divergences[:100]


def analyze_divergent_incentives(
    case,
    sim_incentives: Dict,
    epoch_idx: int,
    tolerance: float,
) -> List[Dict[str, Any]]:
    divergences: List[Dict[str, Any]] = []
    if epoch_idx >= len(case.metas):
        return divergences
    epoch_meta = case.metas[epoch_idx]
    real_incentives = epoch_meta.get("incentives", None)
    if real_incentives is None or not isinstance(real_incentives, torch.Tensor):
        return divergences
    miner_uids = (
        case.miner_indices_epochs[epoch_idx]
        if epoch_idx < len(case.miner_indices_epochs)
        else []
    )
    hotkeys = epoch_meta.get("hotkeys", [])
    for miner_idx, miner_uid in enumerate(miner_uids):
        if miner_idx >= len(hotkeys):
            break
        miner_hotkey = hotkeys[miner_idx]
        sim_incentive = 0.0
        if (
            miner_hotkey in sim_incentives
            and isinstance(sim_incentives[miner_hotkey], list)
            and len(sim_incentives[miner_hotkey]) > 0
        ):
            sim_incentive = float(sim_incentives[miner_hotkey][-1])
        real_incentive = (
            float(real_incentives[miner_uid])
            if miner_uid < real_incentives.shape[0]
            else 0.0
        )
        incentive_diff = abs(sim_incentive - real_incentive)
        if incentive_diff > tolerance:
            divergences.append(
                {
                    "miner_index": miner_idx,
                    "miner_uid": miner_uid,
                    "miner_hotkey": miner_hotkey[:10] + "..."
                    if len(miner_hotkey) > 10
                    else miner_hotkey,
                    "incentive_comparison": {
                        "simulated": sim_incentive,
                        "expected": real_incentive,
                        "difference": incentive_diff,
                    },
                }
            )
    divergences.sort(key=lambda x: x["incentive_comparison"]["difference"], reverse=True)
    return divergences[:100]


def log_incentive_comparison_details(incentive_divergences: List[Dict[str, Any]]) -> None:
    if not incentive_divergences:
        return
    logger.info("=== INCENTIVE COMPARISON DETAILS (Divergences only) ===")
    total_divergences = len(incentive_divergences)
    logger.info(f"Total incentive divergences: {total_divergences}")
    logger.info("Top 10 miners with largest incentive differences:")
    for div in incentive_divergences[:10]:
        miner_uid = div["miner_uid"]
        comp = div["incentive_comparison"]
        sim_val = comp["simulated"]
        real_val = comp["expected"]
        diff = comp["difference"]
        logger.info(
            f"  Miner [UID{miner_uid}]: Sim={sim_val:.6f}, Real={real_val:.6f}, Diff={diff:.6f}"
        )
    if total_divergences > 0:
        max_diff = incentive_divergences[0]["incentive_comparison"]["difference"]
        logger.info(f"Largest incentive difference: {max_diff}")


def save_diagnostic_artifacts(
    artifact_dir: Path, diagnostics: Dict[str, Any], netuid: int
) -> None:
    json_path = artifact_dir / "diagnostic_report.json"
    with open(json_path, "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)
    logger.info(f"Diagnostic JSON report saved to: {json_path}")

    if "weight_analysis_report" in diagnostics:
        weight_file = artifact_dir / "weight_analysis_report.json"
        with open(weight_file, "w") as f:
            json.dump(diagnostics["weight_analysis_report"], f, indent=2, default=str)
        logger.info(f"Weight analysis report saved to: {weight_file}")

    summary_path = artifact_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("VALIDATION FAILURE DIAGNOSTIC SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Network UID: {netuid}\n")
        f.write(
            f"Starting Block: {diagnostics['metadata'].get('starting_block')}\n"
        )
        f.write(
            f"Blocks Tested (anchors): {diagnostics['metadata'].get('blocks_tested')}\n"
        )
        f.write(
            f"Epoch Blocks (used): {diagnostics['metadata'].get('epoch_blocks')}\n"
        )
        f.write(f"Tolerance: {diagnostics['metadata'].get('tolerance_used')}\n\n")

        summary = diagnostics["summary"]
        f.write("VALIDATION STATUS:\n")
        f.write(
            f"  Bonds Match: {'✓' if summary.get('bonds_match') else '✗'}"
        )
        if not summary.get("bonds_match"):
            f.write(
                f" (max diff: {summary.get('bond_max_diff', 0):.6f})"
            )
        f.write("\n")
        f.write(
            f"  Dividends Match: {'✓' if summary.get('dividends_match') else '✗'}"
        )
        if not summary.get("dividends_match"):
            f.write(
                f" (max diff: {summary.get('dividend_max_diff', 0):.6f})"
            )
        f.write("\n")
        f.write(
            f"  Incentives Match: {'✓' if summary.get('incentives_match') else '✗'}"
        )
        if not summary.get("incentive_match"):
            f.write(
                f" (max diff: {summary.get('incentive_max_diff', 0):.6f})"
            )
        f.write("\n\n")
        f.write(f"Blocks tested (anchors): {summary.get('blocks_tested', [])}\n")
        f.write(
            f"Epoch blocks (used): {diagnostics['metadata'].get('epoch_blocks', [])}\n"
        )
        f.write(f"Epochs tested: {summary.get('epochs_tested', 0)}\n\n")

        if diagnostics.get("bond_divergences"):
            f.write(
                f"BOND DIVERGENCES WITH WEIGHT ANALYSIS (showing {min(10, len(diagnostics['bond_divergences']))} of {len(diagnostics['bond_divergences'])}):\n"
            )
            f.write("-" * 80 + "\n")
            for i, div in enumerate(diagnostics["bond_divergences"][:10]):
                f.write(
                    f"\n{i+1}. Validator UID {div['validator_uid']} ({div['validator_hotkey']}) → Target UID {div['target_uid']}\n"
                )
                bond = div["bond_comparison"]
                f.write("   BOND COMPARISON:\n")
                f.write(f"      Simulated: {bond['simulated']:.6f}\n")
                f.write(f"      Expected:  {bond['expected']:.6f}\n")
                f.write(f"      Difference: {bond['difference']:.6f}\n")
                weight = div["weight_comparison"]
                f.write("   WEIGHT COMPARISON:\n")
                f.write(
                    f"      Real (from dumper):  {weight['real_from_dumper']:.6f}\n"
                )
                f.write(
                    f"      Simulator-fed:       {weight['simulator_fed']:.6f}\n"
                )
                f.write(f"      Difference:          {weight['difference']:.6f}\n")

        if diagnostics.get("dividend_divergences"):
            f.write(
                f"\nDIVIDEND DIVERGENCES WITH STAKE ANALYSIS (showing {min(10, len(diagnostics['dividend_divergences']))} of {len(diagnostics['dividend_divergences'])}):\n"
            )
            f.write("-" * 80 + "\n")
            for i, div in enumerate(diagnostics["dividend_divergences"][:10]):
                f.write(
                    f"\n{i+1}. Validator UID {div['validator_uid']} ({div['validator_hotkey']}) - Index {div['validator_index']}\n"
                )
                dividend = div["dividend_comparison"]
                f.write("   DIVIDEND COMPARISON:\n")
                f.write(f"      Simulated: {dividend['simulated']:.6f}\n")
                f.write(f"      Expected:  {dividend['expected']:.6f}\n")
                f.write(f"      Difference: {dividend['difference']:.6f}\n")
                stake = div["stake_comparison"]
                f.write("   STAKE COMPARISON:\n")
                f.write(
                    f"      Real (from dumper):  {stake['real_from_dumper']:.6f}\n"
                )
                f.write(
                    f"      Simulator-fed:       {stake['simulator_fed']:.6f}\n"
                )
                f.write(f"      Difference:          {stake['difference']:.6f}\n")

        if diagnostics.get("incentive_divergences"):
            f.write("\nINCENTIVE DIVERGENCES ANALYSIS:\n")
            f.write("-" * 80 + "\n")
            incentive_divs = diagnostics["incentive_divergences"][:10]
            f.write(
                f"Showing {min(10, len(diagnostics.get('incentive_divergences', [])))} of {len(diagnostics.get('incentive_divergences', []))} incentive divergences:\n\n"
            )
            for i, div in enumerate(incentive_divs):
                comp = div.get("incentive_comparison", {})
                f.write(
                    f"{i+1}. Miner UID {div.get('miner_uid')} → Incentive Mismatch\n"
                )
                f.write(
                    f"   Simulated: {comp.get('simulated', 0):.6f}\n   Expected:  {comp.get('expected', 0):.6f}\n   Difference: {comp.get('difference', 0):.6f}\n\n"
                )
        f.write(f"\nFull diagnostic data saved in: {artifact_dir}\n")
    logger.info(f"Human-readable summary saved to: {summary_path}")
    logger.info(f"All diagnostic artifacts saved to: {artifact_dir}")


def create_weight_analysis_report(case, target_miner_uid: int, netuid: int) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "metadata": {
            "target_miner_uid": target_miner_uid,
            "netuid": netuid,
            "blocks": [case.metas[i].get("block", f"epoch_{i}") for i in range(len(case.metas))],
            "analysis_purpose": "Debug incentive divergence by analyzing validator weights to target miner",
        },
        "epochs": [],
    }
    for epoch_idx in range(len(case.metas)):
        epoch_meta = case.metas[epoch_idx]
        W_full = epoch_meta.get("W", None)
        if W_full is None or not isinstance(W_full, torch.Tensor):
            continue
        hotkeys = epoch_meta.get("hotkeys", [])
        if target_miner_uid >= len(hotkeys):
            continue
        target_hotkey = hotkeys[target_miner_uid]
        if target_hotkey is None:
            logger.warning(
                f"Target miner UID {target_miner_uid} has None hotkey in epoch {epoch_idx}, skipping"
            )
            continue
        validator_uids = (
            case.valid_indices_epochs[epoch_idx]
            if epoch_idx < len(case.valid_indices_epochs)
            else []
        )
        epoch_data: Dict[str, Any] = {
            "epoch": epoch_idx,
            "block": epoch_meta.get("block", f"epoch_{epoch_idx}"),
            "target_miner_hotkey": target_hotkey[:20] + "..."
            if len(target_hotkey) > 20
            else target_hotkey,
            "validators": {},
            "weight_statistics": {
                "total_weight_to_target": 0.0,
                "num_validators_setting_weight": 0,
                "max_weight": 0.0,
                "min_nonzero_weight": float("inf"),
                "weight_distribution": [],
            },
        }
        total_weight = 0.0
        validators_with_weight = 0
        all_weights: List[float] = []
        for validator_uid in validator_uids:
            if validator_uid >= W_full.shape[0]:
                continue
            validator_hotkey = hotkeys[validator_uid] if validator_uid < len(hotkeys) else f"UID{validator_uid}"
            validator_hotkey_short = (
                validator_hotkey[:20] + "..." if len(validator_hotkey) > 20 else validator_hotkey
            )
            weight_to_target = float(W_full[validator_uid, target_miner_uid])
            S = epoch_meta.get("S", torch.zeros(256))
            stake = float(S[validator_uid]) if validator_uid < S.shape[0] else 0.0
            total_outgoing = float(W_full[validator_uid, :].sum())
            epoch_data["validators"][f"uid_{validator_uid}"] = {
                "validator_uid": validator_uid,
                "validator_hotkey": validator_hotkey_short,
                "weight_to_target": weight_to_target,
                "weight_percentage_of_validator": (weight_to_target / total_outgoing * 100)
                if total_outgoing > 0
                else 0.0,
                "validator_stake": stake,
                "validator_total_outgoing_weights": total_outgoing,
            }
            total_weight += weight_to_target
            if weight_to_target > 0:
                validators_with_weight += 1
                all_weights.append(weight_to_target)
                epoch_data["weight_statistics"]["max_weight"] = max(
                    epoch_data["weight_statistics"]["max_weight"], weight_to_target
                )
                if weight_to_target < epoch_data["weight_statistics"]["min_nonzero_weight"]:
                    epoch_data["weight_statistics"]["min_nonzero_weight"] = weight_to_target
        epoch_data["weight_statistics"]["total_weight_to_target"] = total_weight
        epoch_data["weight_statistics"]["num_validators_setting_weight"] = validators_with_weight
        epoch_data["weight_statistics"]["weight_distribution"] = sorted(all_weights, reverse=True)
        if epoch_data["weight_statistics"]["min_nonzero_weight"] == float("inf"):
            epoch_data["weight_statistics"]["min_nonzero_weight"] = 0.0
        if all_weights:
            epoch_data["weight_statistics"]["mean_weight"] = sum(all_weights) / len(all_weights)
            epoch_data["weight_statistics"]["median_weight"] = sorted(all_weights)[len(all_weights) // 2]
        else:
            epoch_data["weight_statistics"]["mean_weight"] = 0.0
            epoch_data["weight_statistics"]["median_weight"] = 0.0
        report["epochs"].append(epoch_data)
    return report


def create_bond_evolution_report(
    case,
    sim_bonds: List[torch.Tensor],
    target_miner_uid: int,
    focus_validator_uid: int,
    netuid: int,
    tested_blocks: List[int],
) -> Dict[str, Any]:
    bond_report: Dict[str, Any] = {
        "metadata": {
            "target_miner_uid": target_miner_uid,
            "focus_validator_uid": focus_validator_uid,
            "netuid": netuid,
            "blocks": tested_blocks,
        },
        "epochs": [],
    }
    for epoch_idx in range(len(case.metas)):
        if epoch_idx >= len(sim_bonds):
            break
        epoch_meta = case.metas[epoch_idx]
        hotkeys = epoch_meta.get("hotkeys", [])
        W_full = epoch_meta.get("W", None)
        if W_full is None or not isinstance(W_full, torch.Tensor):
            continue
        if target_miner_uid >= len(hotkeys):
            continue
        target_hotkey = hotkeys[target_miner_uid]
        validator_uids = (
            case.valid_indices_epochs[epoch_idx]
            if epoch_idx < len(case.valid_indices_epochs)
            else []
        )
        epoch_data: Dict[str, Any] = {
            "epoch": epoch_idx,
            "block": epoch_meta.get("block", f"epoch_{epoch_idx}"),
            "target_miner_hotkey": (
                (target_hotkey[:20] + "...") if isinstance(target_hotkey, str) and len(target_hotkey) > 20 else (target_hotkey if isinstance(target_hotkey, str) else f"UID{target_miner_uid}")
            ),
            "validators": {},
            "summary": {},
        }
        sim_bonds_epoch = sim_bonds[epoch_idx]
        sim_bonds_to_target_sum = 0.0
        sim_bonds_to_target_nonzero = 0
        for validator_uid in validator_uids:
            validator_hotkey = hotkeys[validator_uid] if validator_uid < len(hotkeys) else f"UID{validator_uid}"
            validator_hotkey_short = (
                validator_hotkey[:20] + "..." if isinstance(validator_hotkey, str) and len(validator_hotkey) > 20 else (validator_hotkey if isinstance(validator_hotkey, str) else f"UID{validator_uid}")
            )
            S = epoch_meta.get("S", torch.zeros(256))
            stake = float(S[validator_uid]) if validator_uid < S.shape[0] else 0.0
            weights_val = float(W_full[validator_uid, target_miner_uid])
            # Map validator UID to row index in sim_bonds (rows correspond to validators present this epoch)
            try:
                valid_uids = case.valid_indices_epochs[epoch_idx] if epoch_idx < len(case.valid_indices_epochs) else []
                row_idx = valid_uids.index(validator_uid) if validator_uid in valid_uids else None
            except Exception:
                row_idx = None
            if row_idx is not None and 0 <= row_idx < sim_bonds_epoch.shape[0]:
                sim_bond = float(sim_bonds_epoch[row_idx, target_miner_uid].item())
            else:
                sim_bond = 0.0
            real_bond_raw = float(epoch_meta.get("bonds", torch.zeros(256, 256))[validator_uid, target_miner_uid].item())
            real_bond_norm = real_bond_raw / 65535.0
            col_sum = (epoch_meta.get("bonds", torch.zeros(256, 256))[:, target_miner_uid] / 65535.0).sum().item()
            real_bond_norm = real_bond_norm / (col_sum + 1e-6) if col_sum > 1e-6 else 0.0
            if sim_bond > 0:
                sim_bonds_to_target_sum += sim_bond
                sim_bonds_to_target_nonzero += 1
            epoch_data["validators"][f"uid_{validator_uid}"] = {
                "validator_uid": validator_uid,
                "validator_hotkey": validator_hotkey_short,
                "stake": stake,
                "weights": {"value": weights_val},
                "bonds": {
                    "simulated": sim_bond,
                    "real_raw": real_bond_raw,
                    "real_normalized": real_bond_norm,
                    "real_col_sum": real_bond_norm,
                },
            }
        epoch_data["summary"] = {
            "sim_bonds_to_target_sum": sim_bonds_to_target_sum,
            "sim_bonds_to_target_nonzero": sim_bonds_to_target_nonzero,
            "num_validators": len(validator_uids),
        }
        bond_report["epochs"].append(epoch_data)
    return bond_report


def create_diagnostic_artifacts(
    netuid: int,
    validation_results: Dict[str, Any],
    case,
    sim_bonds: List[torch.Tensor],
    sim_dividends: Dict,
    sim_incentives: Dict,
    tolerance: float,
    timestamp: Optional[str] = None,
    yuma_config: Optional[Any] = None,
) -> Dict[str, Any]:
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = Path(f"validation_artifacts/{timestamp}_subnet_{netuid}")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    starting_block = validation_results.get("blocks_tested", [None])[0]
    blocks_tested = validation_results.get("blocks_tested", [])
    # Per-epoch blocks actually used by the case (preferred for DB lookups)
    epoch_blocks = [case.metas[i].get("block", f"epoch_{i}") for i in range(len(case.metas))]
    diagnostics: Dict[str, Any] = {
        "metadata": {
            "netuid": netuid,
            "starting_block": starting_block,
            "blocks_tested": blocks_tested,
            "epoch_blocks": epoch_blocks,
            "tolerance_used": tolerance,
        },
        "config": {
            "bond_penalty": getattr(yuma_config, "bond_penalty", None)
            if yuma_config
            else None,
            "bonds_moving_avg": getattr(yuma_config, "bond_moving_avg", None)
            if yuma_config
            else None,
            "liquid_alpha_enabled": getattr(yuma_config, "liquid_alpha_enabled", None)
            if yuma_config
            else None,
            "commit_reveal_weights_enabled": getattr(
                yuma_config, "commit_reveal_weights_enabled", None
            )
            if yuma_config
            else None,
            "alpha_high": getattr(yuma_config, "alpha_high", None)
            if yuma_config
            else None,
            "alpha_low": getattr(yuma_config, "alpha_low", None)
            if yuma_config
            else None,
            "kappa": getattr(yuma_config, "kappa", None) if yuma_config else None,
            "liquid_alpha": getattr(yuma_config, "liquid_alpha", None)
            if yuma_config
            else None,
            "rho": getattr(yuma_config, "rho", None) if yuma_config else None,
            "max_weight_limit": getattr(yuma_config, "max_weight_limit", None)
            if yuma_config
            else None,
        },
        "bond_divergences": [],
        "dividend_divergences": [],
        "summary": {},
    }

    last_epoch_idx = validation_results.get("epochs_tested", 0)
    comparisons = validation_results.get("comparisons", {})
    epoch_key = f"epoch_{last_epoch_idx}"
    if epoch_key not in comparisons:
        logger.warning(f"No comparison data for {epoch_key}")
        return diagnostics
    epoch_data = comparisons[epoch_key]
    diagnostics["summary"] = {
        "validation_failed": True,
        "blocks_tested": validation_results.get("blocks_tested", []),
        "epochs_tested": last_epoch_idx,
        "bonds_match": epoch_data.get("bonds", {}).get("matches", False),
        "dividends_match": epoch_data.get("dividends", {}).get("matches", False),
        "incentives_match": epoch_data.get("incentives", {}).get("matches", False),
        "bond_max_diff": epoch_data.get("bonds", {}).get("max_diff", 0),
        "dividend_max_diff": epoch_data.get("dividends", {}).get("max_diff", 0),
        "incentive_max_diff": epoch_data.get("incentives", {}).get("max_diff", 0),
    }

    if not epoch_data.get("bonds", {}).get("matches", False):
        bond_divergences = analyze_divergent_bonds_with_weights(
            case, sim_bonds, last_epoch_idx, tolerance
        )
        diagnostics["bond_divergences"] = bond_divergences
        if bond_divergences:
            max_div = bond_divergences[0]
            max_validator_uid = max_div["validator_uid"]
            max_target_uid = max_div["target_uid"]
            bond_report = create_bond_evolution_report(
                case=case,
                sim_bonds=sim_bonds,
                target_miner_uid=max_target_uid,
                focus_validator_uid=max_validator_uid,
                netuid=netuid,
                tested_blocks=blocks_tested,
            )
            bond_report_path = artifact_dir / "bond_evolution_report.json"
            with open(bond_report_path, "w") as f:
                json.dump(bond_report, f, indent=2)
            summary_path = artifact_dir / "bond_evolution_summary.txt"
            with open(summary_path, "w") as f:
                f.write("BOND EVOLUTION TRACKING REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Target Miner: UID {max_target_uid}\n")
                f.write(f"Focus Validator: UID {max_validator_uid}\n")
                f.write(f"Network: {netuid}\n\n")
                for ep in bond_report["epochs"]:
                    f.write(f"\n--- EPOCH {ep['epoch']} ---\n\n")
                    focus_key = f"uid_{max_validator_uid}"
                    if focus_key in ep["validators"]:
                        focus_val = ep["validators"][focus_key]
                        f.write(f"Focus Validator (UID {max_validator_uid}):\n")
                        f.write(f"  Weight: {focus_val['weights']['value']}\n")
                        f.write(f"  Stake: {focus_val['stake']}\n")
                        f.write(f"  Bonds:\n")
                        f.write(f"    Simulated: {focus_val['bonds']['simulated']}\n")
                        f.write(f"    Real (raw): {focus_val['bonds']['real_raw']}\n")
                        f.write(
                            f"    Real (normalized): {focus_val['bonds']['real_normalized']}\n"
                        )
                        f.write(
                            f"    Real (col sum): {focus_val['bonds']['real_col_sum']}\n\n"
                        )
                    f.write(f"All validators → UID {max_target_uid}:\n")
                    for _, val_data in ep["validators"].items():
                        uid = val_data["validator_uid"]
                        if uid != max_validator_uid:
                            sim_bond = val_data["bonds"]["simulated"]
                            real_bond = val_data["bonds"]["real_col_sum"]
                            weight = val_data["weights"]["value"]
                            f.write(
                                f"  UID {uid:3d}: sim={sim_bond:7.4f}, real={real_bond:7.4f}, weight={weight}\n"
                            )
                    f.write("\nSummary:\n")
                    f.write(
                        f"  Total sim bonds to target: {ep['summary']['sim_bonds_to_target_sum']}\n"
                    )
                    f.write(
                        f"  Non-zero bonds to target: {ep['summary']['sim_bonds_to_target_nonzero']}\n"
                    )

    if not epoch_data.get("dividends", {}).get("matches", False):
        dividend_divergences = analyze_divergent_dividends_with_stakes(
            case, sim_dividends, last_epoch_idx, tolerance
        )
        diagnostics["dividend_divergences"] = dividend_divergences

    incentive_divergences: List[Dict[str, Any]] = []
    if not epoch_data.get("incentives", {}).get("matches", False):
        incentive_divergences = analyze_divergent_incentives(
            case, sim_incentives, last_epoch_idx, tolerance
        )
        diagnostics["incentive_divergences"] = incentive_divergences
        log_incentive_comparison_details(incentive_divergences)

    if incentive_divergences:
        most_divergent = incentive_divergences[0]
        divergent_miner_uid = most_divergent["miner_uid"]
        weight_report = create_weight_analysis_report(
            case=case, target_miner_uid=divergent_miner_uid, netuid=netuid
        )
        diagnostics["weight_analysis_report"] = weight_report

    save_diagnostic_artifacts(artifact_dir, diagnostics, netuid)
    diagnostics["artifact_path"] = str(artifact_dir)
    return diagnostics
