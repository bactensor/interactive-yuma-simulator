#!/usr/bin/env python3
"""
Functional refactor of validation script to compare Yuma simulator output with real metagraph data.

This script:
1. Fetches 2 epochs of real metagraph data (including bonds, dividends, incentives)
2. Runs the YUMA2 simulation on this data
3. Compares simulator output with the real consensus data
4. Reports differences to validate simulator accuracy

The goal is to ensure our simulator matches the currently deployed Yuma consensus.
"""

import os
import sys
import json
import logging
import torch
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

# Add the Django app to the path
sys.path.insert(0, '/root/repos/interactive-yuma-simulator/app/src')

import django
from django.conf import settings

# Configure Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
django.setup()

# Now we can import Django modules
from project.core.utils import fetch_metagraph_weights_stakes, fetch_metagraph_rewards
from project.yuma_simulation._internal.cases import MetagraphCase
from project.yuma_simulation._internal.simulation_utils import _run_dynamic_simulation
from project.yuma_simulation._internal.yumas import YumaConfig, YumaParams, SimulationHyperparameters, YumaSimulationNames, YumaSubtensor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_diagnostic_artifacts(
    netuid: int,
    validation_results: Dict[str, Any],
    case: 'MetagraphCase',
    sim_bonds: List[torch.Tensor],
    sim_dividends: Dict,
    sim_incentives: Dict,
    tolerance: float,
    timestamp: str = None,
    yuma_config: Optional[YumaConfig] = None,
) -> Dict[str, Any]:
    """
    Create diagnostic artifacts for validation failures in functional style.
    
    Args:
        netuid: Network UID being validated
        validation_results: Results from validation
        case: MetagraphCase used in validation
        sim_bonds: Simulated bonds
        sim_dividends: Simulated dividends
        sim_incentives: Simulated incentives
        tolerance: Tolerance used in validation
        timestamp: Optional timestamp for the artifact
        
    Returns:
        Dict containing diagnostic information and file paths
    """
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = Path(f"validation_artifacts/{timestamp}_subnet_{netuid}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract starting block from validation results for metadata
    starting_block = validation_results.get('blocks_tested', [None])[0] 
    blocks_tested = validation_results.get('blocks_tested', [])
    
    diagnostics = {
        "metadata": {
            "netuid": netuid,
            "starting_block": starting_block,
            "blocks_tested": blocks_tested, 
            "tolerance_used": tolerance
        },
        "config": {
            "bond_penalty": getattr(yuma_config, "bond_penalty", None) if yuma_config else None,
            "bonds_moving_avg": getattr(yuma_config, "bond_moving_avg", None) if yuma_config else None,
            "liquid_alpha_enabled": getattr(yuma_config, "liquid_alpha_enabled", None) if yuma_config else None,
            "commit_reveal_weights_enabled": getattr(yuma_config, "commit_reveal_weights_enabled", None) if yuma_config else None,
            "alpha_high": getattr(yuma_config, "alpha_high", None) if yuma_config else None,
            "alpha_low": getattr(yuma_config, "alpha_low", None) if yuma_config else None,
            "kappa": getattr(yuma_config, "kappa", None) if yuma_config else None,
            "liquid_alpha": getattr(yuma_config, "liquid_alpha", None) if yuma_config else None,
            "rho": getattr(yuma_config, "rho", None) if yuma_config else None,
            "max_weight_limit": getattr(yuma_config, "max_weight_limit", None) if yuma_config else None,
        },
        "bond_divergences": [],
        "dividend_divergences": [],
        "summary": {}
    }
    
    # Extract comparison data
    last_epoch_idx = validation_results.get('epochs_tested', 0)
    comparisons = validation_results.get('comparisons', {})
    epoch_key = f'epoch_{last_epoch_idx}'
    
    if epoch_key not in comparisons:
        logger.warning(f"No comparison data for {epoch_key}")
        return diagnostics
    
    epoch_data = comparisons[epoch_key]
    
    # Record summary
    diagnostics["summary"] = {
        "validation_failed": True,
        "blocks_tested": validation_results.get('blocks_tested', []),
        "epochs_tested": last_epoch_idx,
        "bonds_match": epoch_data.get('bonds', {}).get('matches', False),
        "dividends_match": epoch_data.get('dividends', {}).get('matches', False),
        "incentives_match": epoch_data.get('incentives', {}).get('matches', False),
        "bond_max_diff": epoch_data.get('bonds', {}).get('max_diff', 0),
        "dividend_max_diff": epoch_data.get('dividends', {}).get('max_diff', 0),
        "incentive_max_diff": epoch_data.get('incentives', {}).get('max_diff', 0)
    }
    
    # Analyze divergent bonds with weight comparisons
    if not epoch_data.get('bonds', {}).get('matches', False):
        bond_divergences = analyze_divergent_bonds_with_weights(
            case, sim_bonds, last_epoch_idx, tolerance
        )
        diagnostics["bond_divergences"] = bond_divergences
        logger.info(f"Found {len(bond_divergences)} bond divergences for analysis")

        # Generate bond evolution report for the most divergent validator-miner pair
        if bond_divergences:
            max_divergence = bond_divergences[0]  # Already sorted by difference
            max_validator_uid = max_divergence['validator_uid']
            max_target_uid = max_divergence['target_uid']
            
            bond_report = create_bond_evolution_report(
                case=case,
                sim_bonds=sim_bonds,
                target_miner_uid=max_target_uid,
                focus_validator_uid=max_validator_uid,
                netuid=netuid,
                tested_blocks=blocks_tested
            )
            
            # Save the bond evolution report
            bond_report_path = artifact_dir / "bond_evolution_report.json"
            with open(bond_report_path, 'w') as f:
                json.dump(bond_report, f, indent=2)
            
            # Generate human-readable summary
            summary_path = artifact_dir / "bond_evolution_summary.txt"
            with open(summary_path, 'w') as f:
                f.write("BOND EVOLUTION TRACKING REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Target Miner: UID {max_target_uid}\n")
                f.write(f"Focus Validator: UID {max_validator_uid}\n")
                f.write(f"Network: {netuid}\n\n")
                
                # Process each epoch
                for epoch_data in bond_report['epochs']:
                    f.write(f"\n--- EPOCH {epoch_data['epoch']} ---\n\n")
                    
                    # Find focus validator data
                    focus_key = f"uid_{max_validator_uid}"
                    if focus_key in epoch_data['validators']:
                        focus_val = epoch_data['validators'][focus_key]
                        f.write(f"Focus Validator (UID {max_validator_uid}):\n")
                        f.write(f"  Weight: {focus_val['weights']['value']}\n")
                        f.write(f"  Stake: {focus_val['stake']}\n")
                        f.write(f"  Bonds:\n")
                        f.write(f"    Simulated: {focus_val['bonds']['simulated']}\n")
                        f.write(f"    Real (raw): {focus_val['bonds']['real_raw']}\n")
                        f.write(f"    Real (normalized): {focus_val['bonds']['real_normalized']}\n")
                        f.write(f"    Real (col sum): {focus_val['bonds']['real_col_sum']}\n\n")
                    
                    # All validators summary
                    f.write(f"All validators → UID {max_target_uid}:\n")
                    for val_key, val_data in epoch_data['validators'].items():
                        uid = val_data['validator_uid']
                        if uid != max_validator_uid:  # Skip focus validator as we already showed it
                            sim_bond = val_data['bonds']['simulated']
                            real_bond = val_data['bonds']['real_col_sum']
                            weight = val_data['weights']['value']
                            f.write(f"  UID {uid:3d}: sim={sim_bond:7.4f}, real={real_bond:7.4f}, weight={weight}\n")
                    
                    # Epoch summary
                    f.write(f"\nSummary:\n")
                    f.write(f"  Total sim bonds to target: {epoch_data['summary']['sim_bonds_to_target_sum']}\n")
                    f.write(f"  Non-zero bonds to target: {epoch_data['summary']['sim_bonds_to_target_nonzero']}\n")
            
            logger.info(f"Generated bond evolution report for validator {max_validator_uid} → miner {max_target_uid}")

    # Analyze divergent dividends with stake comparisons  
    if not epoch_data.get('dividends', {}).get('matches', False):
        dividend_divergences = analyze_divergent_dividends_with_stakes(
            case, sim_dividends, last_epoch_idx, tolerance
        )
        diagnostics["dividend_divergences"] = dividend_divergences
        logger.info(f"Found {len(dividend_divergences)} dividend divergences for analysis")
    
    # Analyze divergent incentives
    incentive_divergences = []
    if not epoch_data.get('incentives', {}).get('matches', False):
        incentive_divergences = analyze_divergent_incentives(
            case, sim_incentives, last_epoch_idx, tolerance
        )
        diagnostics["incentive_divergences"] = incentive_divergences
        logger.info(f"Found {len(incentive_divergences)} incentive divergences for analysis")
        log_incentive_comparison_details(incentive_divergences)
    
    # Generate weight analysis report for the most divergent incentive miner
    if incentive_divergences:
        most_divergent_incentive = incentive_divergences[0]  # Already sorted by difference
        divergent_miner_uid = most_divergent_incentive['miner_uid']
        
        weight_report = create_weight_analysis_report(
            case=case,
            target_miner_uid=divergent_miner_uid,
            netuid=netuid
        )
        diagnostics["weight_analysis_report"] = weight_report
        logger.info(f"Generated weight analysis report for most divergent incentive miner UID {divergent_miner_uid}")
    
    # Save artifacts
    save_diagnostic_artifacts(artifact_dir, diagnostics, netuid)
    
    diagnostics["artifact_path"] = str(artifact_dir)
    return diagnostics


def analyze_divergent_bonds_with_weights(
    case: 'MetagraphCase',
    sim_bonds: List[torch.Tensor],
    epoch_idx: int,
    tolerance: float
) -> List[Dict[str, Any]]:
    """
    For bonds that diverge beyond tolerance, compare the real weights (from dumper) 
    vs simulator-fed weights to identify data processing issues.
    
    Args:
        case: MetagraphCase with weight and bond data
        sim_bonds: Simulated bonds
        epoch_idx: Epoch index for comparison
        tolerance: Tolerance threshold
        
    Returns:
        List of divergence records with weight comparisons
    """
    divergences = []
    
    if not case.weights_epochs or len(case.weights_epochs) == 0:
        logger.warning("No weight data available for divergence analysis")
        return divergences
    
    # Get data
    simulator_fed_weights = case.weights_epochs[0]  # What was fed to simulator [validators, 256]
    real_bonds = case.metas[epoch_idx].get("bonds", None) if epoch_idx < len(case.metas) else None
    
    # Get original weights from dumper (epoch 0)
    real_weights_full = None
    if len(case.metas) > 0 and "W" in case.metas[0]:
        real_weights_full = case.metas[0]["W"]  # Full [256, 256] matrix from dumper
    
    if real_bonds is None or len(sim_bonds) == 0:
        return divergences
    
    validators = case.validators_epochs[epoch_idx] if len(case.validators_epochs) > epoch_idx else []
    hotkeys = case.metas[epoch_idx]["hotkeys"] if epoch_idx < len(case.metas) else []
    
    # Map validators to UIDs
    validator_positions = {}
    for i, validator_hotkey in enumerate(validators):
        try:
            uid = hotkeys.index(validator_hotkey)
            validator_positions[i] = uid
        except ValueError:
            continue
    
    # Use last simulation epoch bonds
    sim_bonds_epoch = sim_bonds[-1] if sim_bonds else None
    if sim_bonds_epoch is None:
        return divergences
    
    # Analyze divergent bonds
    for val_idx, validator_uid in validator_positions.items():
        if val_idx >= len(simulator_fed_weights) or val_idx >= sim_bonds_epoch.shape[0]:
            continue
        
        validator_hotkey = validators[val_idx]
        
        # Check each target
        for target_uid in range(256):
            # Get bond values
            sim_bond = sim_bonds_epoch[val_idx, target_uid].item()
            real_bond_raw = real_bonds[validator_uid, target_uid].item()
            real_bond_normalized = real_bond_raw / 65535.0
            
            # Apply column normalization
            real_bonds_col_sum = (real_bonds[:, target_uid] / 65535.0).sum().item()
            real_bond_normalized = real_bond_normalized / (real_bonds_col_sum + 1e-6) if real_bonds_col_sum > 1e-6 else 0
            
            bond_diff = abs(sim_bond - real_bond_normalized)
            
            # Only process if bond diverges beyond tolerance
            if bond_diff > tolerance:
                # Get weight comparisons
                simulator_fed_weight = simulator_fed_weights[val_idx][target_uid] if val_idx < len(simulator_fed_weights) else 0
                real_weight_from_dumper = real_weights_full[validator_uid, target_uid].item() if real_weights_full is not None else 0
                
                weight_diff = abs(simulator_fed_weight - real_weight_from_dumper)
                
                divergence = {
                    "validator_uid": validator_uid,
                    "validator_hotkey": validator_hotkey[:10] + "...",
                    "target_uid": target_uid,
                    
                    # Bond comparison (what failed)
                    "bond_comparison": {
                        "simulated": float(sim_bond),
                        "expected": float(real_bond_normalized),
                        "difference": float(bond_diff),
                        "raw_expected": float(real_bond_raw)
                    },
                    
                    # Weight comparison (potential cause)
                    "weight_comparison": {
                        "real_from_dumper": float(real_weight_from_dumper),
                        "simulator_fed": float(simulator_fed_weight),
                        "difference": float(weight_diff)
                    }
                }
                divergences.append(divergence)
    
    # Sort by bond difference (largest first) and limit to top 100
    divergences.sort(key=lambda x: x["bond_comparison"]["difference"], reverse=True)
    return divergences


def analyze_divergent_dividends_with_stakes(
    case: 'MetagraphCase',
    sim_dividends: Dict,
    epoch_idx: int,
    tolerance: float
) -> List[Dict[str, Any]]:
    """
    For dividends that diverge beyond tolerance, compare the real stakes (from dumper)
    vs simulator-fed stakes to identify data processing issues.
    
    Args:
        case: MetagraphCase with stake and dividend data
        sim_dividends: Simulated dividends
        epoch_idx: Epoch index for comparison
        tolerance: Tolerance threshold
        
    Returns:
        List of divergence records with stake comparisons
    """
    divergences = []
    
    if not case.stakes_epochs or len(case.stakes_epochs) == 0:
        logger.warning("No stake data available for divergence analysis")
        return divergences
    
    # Get data
    simulator_fed_stakes = case.stakes_epochs[0]  # What was fed to simulator [validators]
    real_dividends = case.dividends_epochs[epoch_idx] if len(case.dividends_epochs) > epoch_idx else None
    
    # Get original stakes from dumper (epoch 0)
    real_stakes_full = None
    if len(case.metas) > 0 and "S" in case.metas[0]:
        real_stakes_full = case.metas[0]["S"]  # Full [256] array from dumper
    
    if real_dividends is None:
        return divergences
    
    validators = case.validators_epochs[epoch_idx] if len(case.validators_epochs) > epoch_idx else []
    hotkeys = case.metas[epoch_idx]["hotkeys"] if epoch_idx < len(case.metas) else []
    
    # Analyze each validator
    for i, validator_hotkey in enumerate(validators):
        if i >= len(simulator_fed_stakes) or i >= len(real_dividends):
            continue
        
        # Get dividend values
        sim_div = 0.0
        if validator_hotkey in sim_dividends:
            div_data = sim_dividends[validator_hotkey]
            if isinstance(div_data, list) and len(div_data) > 0:
                sim_div = div_data[-1]
            elif isinstance(div_data, (int, float)):
                sim_div = float(div_data)
        
        real_div = real_dividends[i].item()
        div_diff = abs(sim_div - real_div)
        
        # Only process if dividend diverges beyond tolerance
        if div_diff > tolerance:
            # Get stake comparisons
            simulator_fed_stake = simulator_fed_stakes[i]
            
            # Find validator UID to get real stake from dumper
            validator_uid = -1
            try:
                validator_uid = hotkeys.index(validator_hotkey)
            except ValueError:
                pass
            
            real_stake_from_dumper = 0
            if real_stakes_full is not None and validator_uid >= 0 and validator_uid < len(real_stakes_full):
                real_stake_from_dumper = real_stakes_full[validator_uid].item()
            
            stake_diff = abs(simulator_fed_stake - real_stake_from_dumper)
            
            divergence = {
                "validator_index": i,
                "validator_uid": validator_uid,
                "validator_hotkey": validator_hotkey[:10] + "...",
                
                # Dividend comparison (what failed)
                "dividend_comparison": {
                    "simulated": float(sim_div),
                    "expected": float(real_div),
                    "difference": float(div_diff)
                },
                
                # Stake comparison (potential cause)
                "stake_comparison": {
                    "real_from_dumper": float(real_stake_from_dumper),
                    "simulator_fed": float(simulator_fed_stake),
                    "difference": float(stake_diff)
                }
            }
            divergences.append(divergence)
    
    # Sort by dividend difference (largest first) and limit to top 100
    divergences.sort(key=lambda x: x["dividend_comparison"]["difference"], reverse=True)
    return divergences[:100]


def analyze_divergent_incentives(
    case: 'MetagraphCase',
    sim_incentives: Dict,
    epoch_idx: int,
    tolerance: float
) -> List[Dict[str, Any]]:
    """
    For incentives that diverge beyond tolerance, analyze the differences between
    simulated and real incentive values to identify consensus discrepancies.
    
    Args:
        case: MetagraphCase with incentive data
        sim_incentives: Simulated incentives per epoch
        epoch_idx: Epoch index for comparison
        tolerance: Tolerance threshold
        
    Returns:
        List of divergence records with incentive comparisons
    """
    divergences = []
    
    # Get real incentives for this epoch
    if epoch_idx >= len(case.metas):
        return divergences
    
    epoch_meta = case.metas[epoch_idx]
    real_incentives = epoch_meta.get("incentives", None)
    
    if real_incentives is None or not isinstance(real_incentives, torch.Tensor):
        return divergences
    
    # Get active miner UIDs for this epoch
    miner_uids = case.miner_indices_epochs[epoch_idx] if epoch_idx < len(case.miner_indices_epochs) else []
    hotkeys = epoch_meta.get("hotkeys", [])
    
    # Compare incentives for each miner
    for miner_idx, miner_uid in enumerate(miner_uids):
        if miner_idx >= len(hotkeys):
            break
            
        miner_hotkey = hotkeys[miner_idx]
        
        # Get simulated incentive from hotkey-based dictionary
        sim_incentive = 0.0
        if miner_hotkey in sim_incentives and isinstance(sim_incentives[miner_hotkey], list) and len(sim_incentives[miner_hotkey]) > 0:
            sim_incentive = float(sim_incentives[miner_hotkey][-1])  # Last epoch value
        
        # Get real incentive from tensor
        real_incentive = float(real_incentives[miner_uid]) if miner_uid < real_incentives.shape[0] else 0.0
        
        # Check if difference exceeds tolerance
        incentive_diff = abs(sim_incentive - real_incentive)
        if incentive_diff > tolerance:
            divergence = {
                "miner_index": miner_idx,
                "miner_uid": miner_uid,
                "miner_hotkey": miner_hotkey[:10] + "..." if len(miner_hotkey) > 10 else miner_hotkey,
                
                # Incentive comparison (what failed)
                "incentive_comparison": {
                    "simulated": sim_incentive,
                    "expected": real_incentive,
                    "difference": incentive_diff
                }
            }
            
            divergences.append(divergence)
    
    # Sort by incentive difference (largest first) and limit to top 100
    divergences.sort(key=lambda x: x["incentive_comparison"]["difference"], reverse=True)
    return divergences[:100]


def log_incentive_comparison_details(incentive_divergences: List[Dict[str, Any]]) -> None:
    """Log detailed comparison for top incentive divergences"""
    if not incentive_divergences:
        return
        
    logger.info("=== INCENTIVE COMPARISON DETAILS (Divergences only) ===")
    total_divergences = len(incentive_divergences)
    logger.info(f"Total incentive divergences: {total_divergences}")
    
    logger.info("Top 10 miners with largest incentive differences:")
    for _, div in enumerate(incentive_divergences[:10]):
        miner_uid = div['miner_uid']
        comp = div['incentive_comparison']
        sim_val = comp['simulated']
        real_val = comp['expected']
        diff = comp['difference']
        
        logger.info(f"  Miner [UID{miner_uid}]: Sim={sim_val:.6f}, Real={real_val:.6f}, Diff={diff:.6f}")
    
    if total_divergences > 0:
        max_diff = incentive_divergences[0]['incentive_comparison']['difference']
        logger.info(f"Largest incentive difference: {max_diff}")

def save_diagnostic_artifacts(
    artifact_dir: Path,
    diagnostics: Dict[str, Any],
    netuid: int
) -> None:
    """
    Save diagnostic artifacts to files.
    
    Args:
        artifact_dir: Directory to save artifacts
        diagnostics: Diagnostic data to save
        netuid: Network UID
    """
    # Save JSON report
    json_path = artifact_dir / "diagnostic_report.json"
    with open(json_path, 'w') as f:
        json.dump(diagnostics, f, indent=2, default=str)
    logger.info(f"Diagnostic JSON report saved to: {json_path}")
    
    # Save weight analysis report separately if it exists
    if "weight_analysis_report" in diagnostics:
        weight_file = artifact_dir / "weight_analysis_report.json"
        with open(weight_file, 'w') as f:
            json.dump(diagnostics["weight_analysis_report"], f, indent=2, default=str)
        logger.info(f"Weight analysis report saved to: {weight_file}")
    
    # Save human-readable summary
    summary_path = artifact_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("VALIDATION FAILURE DIAGNOSTIC SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Network UID: {netuid}\n")
        f.write(f"Starting Block: {diagnostics['metadata']['starting_block']}\n")
        f.write(f"Blocks Tested: {diagnostics['metadata']['blocks_tested']}\n")
        f.write(f"Tolerance: {diagnostics['metadata']['tolerance_used']}\n\n")
        
        summary = diagnostics['summary']
        f.write("VALIDATION STATUS:\n")
        f.write(f"  Bonds Match: {'✓' if summary.get('bonds_match') else '✗'}")
        if not summary.get('bonds_match'):
            f.write(f" (max diff: {summary.get('bond_max_diff', 0):.6f})")
        f.write("\n")
        
        f.write(f"  Dividends Match: {'✓' if summary.get('dividends_match') else '✗'}")
        if not summary.get('dividends_match'):
            f.write(f" (max diff: {summary.get('dividend_max_diff', 0):.6f})")
        f.write("\n")
        
        f.write(f"  Incentives Match: {'✓' if summary.get('incentives_match') else '✗'}")
        if not summary.get('incentive_match'):
            f.write(f" (max diff: {summary.get('incentive_max_diff', 0):.6f})")
        f.write("\n\n")
        
        f.write(f"Blocks tested: {summary.get('blocks_tested', [])}\n")
        f.write(f"Epochs tested: {summary.get('epochs_tested', 0)}\n\n")
        
        # Write bond divergences with weight analysis
        if diagnostics["bond_divergences"]:
            f.write(f"BOND DIVERGENCES WITH WEIGHT ANALYSIS (showing {min(10, len(diagnostics['bond_divergences']))} of {len(diagnostics['bond_divergences'])}):\n")
            f.write("-" * 80 + "\n")
            
            
            for i, div in enumerate(diagnostics["bond_divergences"][:10]):
                f.write(f"\n{i+1}. Validator UID {div['validator_uid']} ({div['validator_hotkey']}) → Target UID {div['target_uid']}\n")
                
                # Bond comparison (what failed)
                bond = div['bond_comparison']
                f.write(f"   📊 BOND COMPARISON:\n")
                f.write(f"      Simulated: {bond['simulated']:.6f}\n")
                f.write(f"      Expected:  {bond['expected']:.6f}\n")
                f.write(f"      Difference: {bond['difference']:.6f}\n")
                
                # Weight comparison (potential cause)
                weight = div['weight_comparison']
                f.write(f"   ⚖️  WEIGHT COMPARISON:\n")
                f.write(f"      Real (from dumper):  {weight['real_from_dumper']:.6f}\n")
                f.write(f"      Simulator-fed:       {weight['simulator_fed']:.6f}\n")
                f.write(f"      Difference:          {weight['difference']:.6f}\n")
            f.write("\n")
        
        # Write dividend divergences with stake analysis
        if diagnostics["dividend_divergences"]:
            f.write(f"DIVIDEND DIVERGENCES WITH STAKE ANALYSIS (showing {min(10, len(diagnostics['dividend_divergences']))} of {len(diagnostics['dividend_divergences'])}):\n")
            f.write("-" * 80 + "\n")
            
            
            for i, div in enumerate(diagnostics["dividend_divergences"][:10]):
                f.write(f"\n{i+1}. Validator UID {div['validator_uid']} ({div['validator_hotkey']}) - Index {div['validator_index']}\n")
                
                # Dividend comparison (what failed)
                dividend = div['dividend_comparison']
                f.write(f"   💰 DIVIDEND COMPARISON:\n")
                f.write(f"      Simulated: {dividend['simulated']:.6f}\n")
                f.write(f"      Expected:  {dividend['expected']:.6f}\n")
                f.write(f"      Difference: {dividend['difference']:.6f}\n")
                
                # Stake comparison (potential cause)
                stake = div['stake_comparison']
                f.write(f"   🏦 STAKE COMPARISON:\n")
                f.write(f"      Real (from dumper):  {stake['real_from_dumper']:.6f}\n")
                f.write(f"      Simulator-fed:       {stake['simulator_fed']:.6f}\n")
                f.write(f"      Difference:          {stake['difference']:.6f}\n")
            f.write("\n")

        # Write incentive divergences
        if diagnostics.get("incentive_divergences"):
            f.write(f"INCENTIVE DIVERGENCES (showing {min(10, len(diagnostics['incentive_divergences']))} of {len(diagnostics['incentive_divergences'])}):\n")
            f.write("-" * 80 + "\n")
            
            for i, inc in enumerate(diagnostics["incentive_divergences"][:10]):
                f.write(f"\n{i+1}. Miner UID {inc['miner_uid']} ({inc['miner_hotkey']}) - Index {inc['miner_index']}\n")
                
                # Incentive comparison
                incentive = inc['incentive_comparison']
                f.write(f"   🎯 INCENTIVE COMPARISON:\n")
                f.write(f"      Simulated: {incentive['simulated']:.6f}\n")
                f.write(f"      Expected:  {incentive['expected']:.6f}\n")
                f.write(f"      Difference: {incentive['difference']:.6f}\n")
            f.write("\n")

        # Forced UID 24 snapshots (weights and bonds)
        if diagnostics.get("selected_miner_weights_uid_24") or diagnostics.get("selected_miner_bonds_uid_24"):
            f.write("FORCED UID 24 SNAPSHOTS:\n")
            f.write("-" * 80 + "\n")
            if diagnostics.get("selected_miner_weights_uid_24"):
                fw = diagnostics["selected_miner_weights_uid_24"]
                f.write(f"  Weights: epoch={fw.get('epoch_idx')}, target_uid={fw.get('target_uid')}, target_hk={fw.get('target_hotkey','')}\n")
            if diagnostics.get("selected_miner_bonds_uid_24"):
                fb = diagnostics["selected_miner_bonds_uid_24"]
                sums = fb.get('sums', {})
                f.write(f"  Bonds:   epoch={fb.get('epoch_idx')}, target_uid={fb.get('target_uid')}, target_hk={fb.get('target_hotkey','')}\n")
                f.write(f"           real_col_sum_pre_norm={sums.get('real_col_sum_pre_norm')}\n")
                f.write(f"           real_col_sum_post_norm={sums.get('real_col_sum_post_norm')}\n")
                f.write(f"           sim_B_col_sum={sums.get('sim_B_col_sum')}\n")
                f.write(f"           sim_B_ema_col_sum={sums.get('sim_B_ema_col_sum')}\n")
                f.write(f"           alpha={sums.get('alpha')}\n\n")

        # Write incentive divergences analysis if they exist
        if diagnostics.get("incentive_divergences"):
            f.write("INCENTIVE DIVERGENCES ANALYSIS:\n")
            f.write("-" * 80 + "\n")
            incentive_divs = diagnostics["incentive_divergences"][:10]  # Show top 10
            f.write(f"Showing {min(10, len(diagnostics.get('incentive_divergences', [])))} of {len(diagnostics.get('incentive_divergences', []))} incentive divergences:\n\n")
            
            for i, div in enumerate(incentive_divs):
                f.write(f"{i+1}. Miner UID {div.get('miner_uid')} → Incentive Mismatch\n")
                f.write(f"   Simulated: {div.get('simulated_incentive', 0):.6f}\n")
                f.write(f"   Expected:  {div.get('expected_incentive', 0):.6f}\n")
                f.write(f"   Difference: {div.get('difference', 0):.6f}\n\n")
        
        f.write(f"\nFull diagnostic data saved in: {artifact_dir}\n")
    
    logger.info(f"Human-readable summary saved to: {summary_path}")
    logger.info(f"📁 All diagnostic artifacts saved to: {artifact_dir}")


def normalize_hyperparameters(raw_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize raw hyperparameters from blockchain format to simulator format.
    
    Args:
        raw_params: Raw hyperparameters from the service
        
    Returns:
        Dict with normalized values for the simulator
        
    Raises:
        KeyError: If required parameters are missing
    """
    # Check for required parameters
    required = ['alpha_low', 'alpha_high', 'kappa', 'bonds_moving_avg', 
               'liquid_alpha_enabled']
    missing = [p for p in required if p not in raw_params]
    if missing:
        available = list(raw_params.keys())
        raise KeyError(
            f"Missing required hyperparameters: {missing}\n"
            f"Available parameters: {available}"
        )
    
    # Normalize values with proper scaling
    normalized = {
        'alpha_low': raw_params['alpha_low'] / 65535.0,
        'alpha_high': raw_params['alpha_high'] / 65535.0,
        'kappa': raw_params['kappa'] / 65535.0,
        'bonds_moving_avg': raw_params['bonds_moving_avg'] / 1000000.0,
        'liquid_alpha_enabled': bool(raw_params['liquid_alpha_enabled']),
    }
    
    # Handle optional parameters
    if 'alpha_sigmoid_steepness' in raw_params:
        normalized['alpha_sigmoid_steepness'] = float(raw_params['alpha_sigmoid_steepness'])
    else:
        normalized['alpha_sigmoid_steepness'] = 10.0
    
    # Determine bond_penalty: prefer explicit field if available; otherwise fallback to alpha_high
    bond_penalty = None
    if 'bond_penalty' in raw_params:
        bp_raw = raw_params['bond_penalty']
        try:
            bp_val = float(bp_raw)
            # Heuristic scaling: if > 1.0, assume 16-bit fixed-point like alphas
            if bp_val > 1.0:
                bp_val = bp_val / 65535.0
            bond_penalty = max(0.0, min(1.0, bp_val))
            logger.info(f"Using bond_penalty from hyperparams: raw={bp_raw}, normalized={bond_penalty}")
        except Exception:
            logger.warning(f"Failed to parse bond_penalty from hyperparams: {bp_raw}")
    if bond_penalty is None:
        bond_penalty = 1.0
    normalized['bond_penalty'] = bond_penalty
    
    return normalized


def get_top_subnets_by_tao_emission(top_n: int = 10) -> List[int]:
    """
    Get top N subnets sorted by tao_in_emission.
    
    Args:
        top_n: Number of top subnets to return
        
    Returns:
        List of netuid integers for top subnets
    """
    try:
        import bittensor as bt
        s = bt.subtensor()
        
        # Get all subnets and sort by tao_in_emission
        all_subnets = list(s.all_subnets())
        all_subnets.sort(key=lambda sn: sn.tao_in_emission, reverse=True)
        
        # Filter out subnet 0 (root network) and get top N
        filtered_subnets = [sn for sn in all_subnets if sn.netuid != 0]
        top_subnets = filtered_subnets[:top_n]
        
        # Return just the netuids
        return [sn.netuid for sn in top_subnets]
    except Exception as e:
        logger.error(f"Error fetching top subnets: {e}")
        return []


def fetch_subnet_hyperparameters(netuid: int, service_url: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Fetch hyperparameters for a specific subnet from the hyperparameters service.
    
    Args:
        netuid: Network UID to fetch hyperparameters for
        service_url: Base URL of the hyperparameters service (defaults to env var)
        
    Returns:
        Dict containing hyperparameters or None if fetch fails
    """
    if service_url is None:
        service_url = os.getenv('HYPERPARAMS_SERVICE_URL', 'http://localhost:8000/')
    
    # Ensure URL ends with /
    if not service_url.endswith('/'):
        service_url += '/'
    
    url = f"{service_url}hyperparams/subnet/{netuid}/"
    
    try:
        logger.info(f"Fetching hyperparameters from {url}")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            yuma_version = "YUMA3" if data.get('is_yuma3_on', False) else "YUMA2"
            logger.info(f"Successfully fetched hyperparameters for subnet {netuid}")
            return data
        elif response.status_code == 404:
            logger.warning(f"No hyperparameters found for subnet {netuid}")
            return None
        else:
            logger.error(f"Failed to fetch hyperparameters: HTTP {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching hyperparameters: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching hyperparameters: {e}")
        return None


def fetch_multiple_subnets_hyperparameters(subnet_ids: List[int], service_url: Optional[str] = None) -> Dict[int, Dict[str, Any]]:
    """
    Fetch hyperparameters for multiple subnets from the hyperparameters service.
    
    Args:
        subnet_ids: List of network UIDs to fetch hyperparameters for
        service_url: Base URL of the hyperparameters service (defaults to env var)
        
    Returns:
        Dict mapping subnet_id to hyperparameters dict
    """
    if service_url is None:
        service_url = os.getenv('HYPERPARAMS_SERVICE_URL', 'http://localhost:8000/')
    
    # Ensure URL ends with /
    if not service_url.endswith('/'):
        service_url += '/'
    
    # Use the multiple subnets endpoint
    subnet_ids_str = ','.join(map(str, subnet_ids))
    url = f"{service_url}hyperparams/multiple/?subnet_ids={subnet_ids_str}"
    
    try:
        logger.info(f"Fetching hyperparameters for subnets {subnet_ids} from {url}")
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Convert to dict[subnet_id] = hyperparams
            result = {}
            for subnet_data in data.get('subnets', []):
                subnet_id = subnet_data['subnet']
                yuma_version = "YUMA3" if subnet_data.get('is_yuma3_on', False) else "YUMA2"
                result[subnet_id] = subnet_data
                logger.info(f"Successfully fetched hyperparameters for subnet {subnet_id} (Yuma version: {yuma_version})")
            
            # Log any subnets that weren't found
            not_found = data.get('not_found', [])
            if not_found:
                logger.warning(f"Hyperparameters not found for subnets: {not_found}")
            
            return result
        elif response.status_code == 404:
            logger.warning(f"No hyperparameters found for any of the subnets {subnet_ids}")
            return {}
        else:
            logger.error(f"Failed to fetch hyperparameters: HTTP {response.status_code}")
            return {}
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching hyperparameters for subnets {subnet_ids}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error fetching hyperparameters for subnets {subnet_ids}: {e}")
        return {}


def prepare_metagraph_data(
    start_date: datetime,
    end_date: datetime,
    netuid: int,
    max_epochs: int = 2
) -> tuple[MetagraphCase, list]:
    """
    Fetch and prepare metagraph data for validation.
    
    Args:
        start_date: Start date for data fetching
        end_date: End date for data fetching
        netuid: Network UID
        max_epochs: Maximum number of epochs to validate (default: 3)
        
    Returns:
        Tuple of (MetagraphCase prepared for validation, list of blocks tested)
        
    Raises:
        ValueError: If insufficient epochs are available
    """
    logger.info(f"Fetching metagraph data for up to {max_epochs} epochs...")
    try:
        # Step 1: Fetch weights/stakes for all epochs (needed for simulation)
        weights_stakes_data = fetch_metagraph_weights_stakes(
            start_date=start_date,
            end_date=end_date,
            netuid=netuid
        )
        
        # Step 2: Get blocks from weights/stakes and determine which rewards to fetch
        blocks = weights_stakes_data.get('blocks', [])
        if len(blocks) < max_epochs:
            logger.warning(f"Only {len(blocks)} blocks available, requested {max_epochs}")
            max_epochs = len(blocks)
        
        if max_epochs < 3:
            raise ValueError(f"Need at least 3 epochs for validation (epoch 0 for B_old, epochs 1+ for validation), got {max_epochs}")
        
        # Step 3: Fetch rewards data including bonds for all epochs
        first_block = blocks[0]
        last_block = blocks[max_epochs - 1]
        
        logger.info(f"Fetching rewards for blocks {first_block} (epoch 0) and {last_block} (epoch {max_epochs-1})")
        
        rewards_data = fetch_metagraph_rewards(
            start_date=start_date,
            end_date=end_date,
            netuid=netuid
        )
        
        mg_data = {**weights_stakes_data, **rewards_data}

    except Exception as e:
        logger.error(f"Failed to fetch metagraph data: {e}")
        raise
    
    try:
        case, _ = MetagraphCase.from_mg_dumper_data(mg_data)
    except Exception as e:
        logger.error(f"Error creating MetagraphCase: {e}")
        raise

    if case.num_epochs < max_epochs:
        logger.info(f"Limiting epochs from {case.num_epochs} to {max_epochs}")
        max_epochs = case.num_epochs
    
    # Limit to requested number of epochs
    case.num_epochs = max_epochs
    case.metas = case.metas[:max_epochs]
        
    # Return limited blocks list for reporting
    tested_blocks = blocks[:max_epochs]
    return case, tested_blocks


def setup_yuma_configuration(netuid: int, bond_penalty_override: Optional[float] = None) -> Tuple[YumaConfig, bool]:
    """
    Fetch hyperparameters and setup Yuma configuration.
    
    Args:
        netuid: Network UID
        bond_penalty_override: Optional override for bond_penalty
        
    Returns:
        Tuple of (YumaConfig object, is_yuma3_on boolean)
        
    Raises:
        ValueError: If hyperparameters cannot be fetched or normalized
    """
    hyperparams_data = fetch_subnet_hyperparameters(netuid)
    
    if not hyperparams_data or 'hyperparams' not in hyperparams_data:
        raise ValueError(f"Failed to fetch hyperparameters for subnet {netuid}. Cannot proceed with validation.")
    
    raw_hyperparams = hyperparams_data['hyperparams']
    is_yuma3_on = hyperparams_data.get('is_yuma3_on', False)
    
    try:
        params = normalize_hyperparameters(raw_hyperparams)
        
        # Determine effective bond_penalty
        effective_bond_penalty = params['bond_penalty']
        if bond_penalty_override is not None:
            logger.info(f"Overriding bond_penalty to {bond_penalty_override} for parity/testing")
            effective_bond_penalty = float(bond_penalty_override)

        yuma_config = YumaConfig(
            simulation=SimulationHyperparameters(
                kappa=params['kappa'],
                bond_penalty=effective_bond_penalty,
                total_epoch_emission=100.0, #TODO: Get from service if available
                total_subnet_stake=1_000_000.0, #TODO: Get from service if available
            ),
            yuma_params=YumaParams(
                bond_moving_avg=params['bonds_moving_avg'],
                liquid_alpha=params['liquid_alpha_enabled'],
                alpha_high=params['alpha_high'],
                alpha_low=params['alpha_low'],
                alpha_sigmoid_steepness=params['alpha_sigmoid_steepness'],
            )
        )
        
        return yuma_config, is_yuma3_on
        
    except KeyError as e:
        logger.error(f"Hyperparameter validation failed: {e}")
        raise ValueError(f"Cannot proceed with validation: {e}")
    except Exception as e:
        logger.error(f"Error processing hyperparameters: {e}")
        raise


def compare_bonds(
    sim_bonds: List[torch.Tensor],
    real_bonds_full: torch.Tensor,
    validators_epoch: List[str],
    epoch_hotkeys: List[str],
    tolerance: float,
    case=None,
    sim_epoch_idx: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compare simulation bonds with real bonds.
    
    Args:
        sim_bonds: List of simulation bond tensors [validators, 256]
        real_bonds_full: Real bonds tensor [256, 256] (full metagraph)
        validators_epoch: List of validator hotkeys for the epoch
        epoch_hotkeys: Full list of hotkeys at all 256 positions
        tolerance: Numerical tolerance for comparison
        
    Returns:
        Dict with comparison results
    """
    if len(sim_bonds) == 0 or real_bonds_full is None:
        return {
            'error': 'Missing bonds data for comparison',
            'matches': False
        }
    
    # Use the correct simulation epoch based on whether epoch 0 bonds were used as B_old
    if sim_epoch_idx is not None:
        if sim_epoch_idx < 0 or sim_epoch_idx >= len(sim_bonds):
            return {
                'error': f'Invalid sim_epoch_idx {sim_epoch_idx} for {len(sim_bonds)} simulation epochs',
                'matches': False
            }
        sim_bonds_epoch = sim_bonds[sim_epoch_idx]  # Shape: [num_validators, 256]
    else:
        # Fallback to last epoch if not specified
        sim_bonds_epoch = sim_bonds[-1]  # Shape: [num_validators, 256]    
    # Extract validator rows from full real bonds matrix
    validator_positions = []
    for i, validator_hotkey in enumerate(validators_epoch):
        try:
            pos = epoch_hotkeys.index(validator_hotkey)
            validator_positions.append(pos)
        except ValueError:
            logger.warning(f"Validator {validator_hotkey[:10]}... not found in epoch hotkeys")
            return {
                'error': f'Validator {validator_hotkey[:10]}... not found in epoch hotkeys',
                'matches': False
            }
    
    # Extract rows for active validators from full real bonds
    real_bonds_epoch = real_bonds_full[validator_positions, :]  # Shape: [num_validators, 256]
        
    if sim_bonds_epoch.shape != real_bonds_epoch.shape:
        return {
            'error': f"Shape mismatch after filtering: sim={sim_bonds_epoch.shape}, real={real_bonds_epoch.shape}",
            'matches': False
        }
    
    # Normalize real bonds from blockchain format (0-65535) to 0-1 range
    real_bonds_normalized = real_bonds_epoch / 65535.0
    
    # Apply the same column-wise normalization that the simulator does
    real_bonds_column_sums = real_bonds_normalized.sum(dim=0)
    real_bonds_normalized = real_bonds_normalized / (real_bonds_column_sums + 1e-6)
    real_bonds_normalized = torch.nan_to_num(real_bonds_normalized)

    # Compare normalized tensors
    bonds_diff = torch.abs(sim_bonds_epoch - real_bonds_normalized)
    bonds_max_diff = torch.max(bonds_diff).item()
    bonds_mean_diff = torch.mean(bonds_diff).item()
    bonds_nonzero_diff = (bonds_diff > tolerance).sum().item()
    
    # Log detailed comparison for non-zero bonds
    logger.info("=== BONDS COMPARISON DETAILS (Non-zero bonds only) ===")
    
    sim_nonzero_mask = sim_bonds_epoch > 0
    real_nonzero_mask = real_bonds_normalized > 0
    any_nonzero_mask = sim_nonzero_mask | real_nonzero_mask
    
    sim_nonzero_count = sim_nonzero_mask.sum().item()
    real_nonzero_count = real_nonzero_mask.sum().item()
    total_nonzero_positions = any_nonzero_mask.sum().item()
    
    logger.info(f"Non-zero bonds - Sim: {sim_nonzero_count}, Real: {real_nonzero_count}, Total positions: {total_nonzero_positions}")
    
    if total_nonzero_positions > 0:
        nonzero_positions = torch.where(any_nonzero_mask)
        
        # Get differences for all positions and sort by largest difference
        position_diffs = []
        for i in range(len(nonzero_positions[0])):
            row, col = nonzero_positions[0][i].item(), nonzero_positions[1][i].item()
            sim_val = sim_bonds_epoch[row, col].item()
            real_val = real_bonds_normalized[row, col].item()
            diff = abs(sim_val - real_val)
            validator_uid = validator_positions[row] if row < len(validator_positions) else row
            position_diffs.append((diff, row, col, sim_val, real_val, validator_uid))
        
        # Sort by difference (descending)
        position_diffs.sort(key=lambda x: x[0], reverse=True)
        
        logger.info("Top 10 positions with largest bond differences:")
        for i in range(min(10, len(position_diffs))):
            diff, row, col, sim_val, real_val, validator_uid = position_diffs[i]
            
            # Get weights and stakes if case data available
            weight_info = ""
            stake_info = ""
            if case is not None and len(case.weights_epochs) > 0:
                try:
                    # Get epoch 0 weights (input epoch)
                    weights_epoch0 = case.weights_epochs[0]  # [validators, 256]
                    if row < len(weights_epoch0) and col < len(weights_epoch0[row]):
                        weight_val = weights_epoch0[row][col]
                        weight_info = f", Weight={weight_val:.6f}"
                except (IndexError, TypeError):
                    weight_info = ", Weight=N/A"
                    
                try:
                    # Get validator stake (epoch 0)
                    stakes_epoch0 = case.stakes_epochs[0]  # [validators]
                    if row < len(stakes_epoch0):
                        stake_val = stakes_epoch0[row]
                        stake_info = f", Stake={stake_val:.6f}"
                except (IndexError, TypeError):
                    stake_info = ", Stake=N/A"
            
            logger.info(f"  Position [UID{validator_uid},{col}]: Sim={sim_val:.6f}, Real={real_val:.6f}, Diff={diff:.6f}{weight_info}{stake_info}")
        
        nonzero_diffs = bonds_diff[any_nonzero_mask]
        if len(nonzero_diffs) > 0:
            max_nonzero_diff = torch.max(nonzero_diffs).item()
            logger.info(f"Largest difference among non-zero positions: {max_nonzero_diff}")
    else:
        logger.info("No non-zero bonds found in either simulation or real data")
    
    return {
        'max_diff': bonds_max_diff,
        'mean_diff': bonds_mean_diff,
        'nonzero_diffs': bonds_nonzero_diff,
        'matches': bonds_max_diff < tolerance,
        'sim_shape': list(sim_bonds_epoch.shape),
        'real_shape': list(real_bonds_normalized.shape),
        'sim_nonzero_bonds': sim_nonzero_count,
        'real_nonzero_bonds': real_nonzero_count,
        'note': 'Direct comparison of filtered bond matrices'
    }


def compare_dividends(
    sim_normalized_dividends: Dict,
    real_dividends_epoch1: torch.Tensor,
    validators_epoch1: List[str],
    tolerance: float
) -> Dict[str, Any]:
    """
    Compare simulation dividends with real dividends (both already filtered).
    
    Args:
        sim_normalized_dividends: Normalized dividends from simulation (dict by hotkey)
        real_dividends_epoch1: Real dividends tensor for epoch 1 (already filtered)
        validators_epoch1: List of validator hotkeys (in same order as real tensor)
        tolerance: Numerical tolerance for comparison
        
    Returns:
        Dict with comparison results
    """
    if real_dividends_epoch1 is None:
        return {
            'error': 'No real dividends data available',
            'matches': False
        }
    
    logger.info(f"=== DIVIDENDS COMPARISON DETAILS ===")
    logger.info(f"Validators count: {len(validators_epoch1)}")
    
    # Extract simulation dividends in the same order as validators_epoch1
    sim_div_values = []
    for i, validator_hotkey in enumerate(validators_epoch1):
        if validator_hotkey in sim_normalized_dividends:
            sim_div_data = sim_normalized_dividends[validator_hotkey]
            if isinstance(sim_div_data, list) and len(sim_div_data) > 0:
                sim_div_values.append(sim_div_data[-1])  # Use last epoch
            elif isinstance(sim_div_data, (int, float)):
                sim_div_values.append(float(sim_div_data))
            else:
                logger.warning(f"Validator {i} has unusual dividend data: {sim_div_data}")
                sim_div_values.append(0.0)
        else:
            logger.warning(f"Validator {i} ({validator_hotkey[:10]}...) not found in sim dividends")
            sim_div_values.append(0.0)
    
    if len(sim_div_values) != len(validators_epoch1):
        return {
            'error': f'Simulation dividends length mismatch: got {len(sim_div_values)}, expected {len(validators_epoch1)}',
            'matches': False
        }
    
    sim_div_tensor = torch.tensor(sim_div_values, dtype=torch.float32)
    
    # Direct comparison - both tensors should have same length now!
    if sim_div_tensor.shape != real_dividends_epoch1.shape:
        return {
            'error': f"Shape mismatch: sim={sim_div_tensor.shape}, real={real_dividends_epoch1.shape}",
            'matches': False
        }
    
    # Handle scaling if needed
    sim_div_scale = sim_div_tensor.max().item()
    real_div_scale = real_dividends_epoch1.max().item()
    scale_ratio = real_div_scale / sim_div_scale if sim_div_scale > 0 else float('inf')
        
    if sim_div_scale < 1e-6 and real_div_scale > 0.01:
        logger.info("Scaling simulation dividends to match real dividends")
        sim_div_tensor = sim_div_tensor * scale_ratio
    elif real_div_scale > 10:
        logger.info("Normalizing real dividends by 65535")
        real_dividends_epoch1 = real_dividends_epoch1 / 65535.0
    
    # Calculate differences
    div_diff = torch.abs(sim_div_tensor - real_dividends_epoch1)
    div_max_diff = torch.max(div_diff).item()
    div_mean_diff = torch.mean(div_diff).item()
    div_nonzero_diff = (div_diff > tolerance).sum().item()
    
    # Log detailed per-validator comparison
    nonzero_mask = (sim_div_tensor > 0) | (real_dividends_epoch1 > 0)
    nonzero_count = nonzero_mask.sum().item()
    logger.info(f"Non-zero dividends: {nonzero_count}")
    
    if nonzero_count > 0:
        logger.info("Dividend comparisons (showing all validators):")
        for i in range(len(validators_epoch1)):
            validator_hotkey = validators_epoch1[i][:10] + "..."
            sim_val = sim_div_tensor[i].item()
            real_val = real_dividends_epoch1[i].item()
            diff = abs(sim_val - real_val)
            logger.info(f"  Validator {validator_hotkey}: Sim={sim_val:.12e}, Real={real_val:.6f}, Diff={diff:.6f}")
    
    logger.info(f"Max difference: {div_max_diff:.6f}")
    logger.info(f"Mean difference: {div_mean_diff:.6f}")
    
    return {
        'max_diff': div_max_diff,
        'mean_diff': div_mean_diff,
        'nonzero_diffs': div_nonzero_diff,
        'matches': div_max_diff < tolerance,
        'sim_nonzero_count': (sim_div_tensor > 0).sum().item(),
        'real_nonzero_count': (real_dividends_epoch1 > 0).sum().item(),
        'note': 'Direct comparison using filtered dividend data'
    }


def compare_incentives(
    sim_incentives_per_epoch: Dict,
    real_incentives_epoch1: torch.Tensor,
    miners: List[str],
    tolerance: float,
    sim_epoch_idx: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compare simulation incentives with real incentives (both already filtered).
    
    Args:
        sim_incentives_per_epoch: Simulation incentives per miner
        real_incentives_epoch1: Real incentives tensor for epoch 1 (already filtered to active miners)
        miners: List of miner hotkeys (in same order as real tensor)
        tolerance: Numerical tolerance for comparison
        
    Returns:
        Dict with comparison results
    """
    if real_incentives_epoch1 is None:
        return {
            'error': 'No real incentives data available',
            'matches': False
        }
    
    # Build simulation incentives tensor using the correct epoch
    sim_inc_values = []
    for miner_hotkey in miners:
        if miner_hotkey in sim_incentives_per_epoch and len(sim_incentives_per_epoch[miner_hotkey]) > 0:
            incentive_list = sim_incentives_per_epoch[miner_hotkey]
            if sim_epoch_idx is not None and sim_epoch_idx < len(incentive_list):
                sim_inc_values.append(incentive_list[sim_epoch_idx])
            else:
                # Fallback to last epoch
                sim_inc_values.append(incentive_list[-1])
        else:
            sim_inc_values.append(0.0)
    
    if not sim_inc_values:
        return {
            'error': 'No simulation incentives data available',
            'matches': False
        }
    
    sim_inc_tensor = torch.tensor(sim_inc_values, dtype=torch.float32)
    
    # Compare tensors
    if sim_inc_tensor.shape != real_incentives_epoch1.shape:
        return {
            'error': f"Shape mismatch: sim={sim_inc_tensor.shape}, real={real_incentives_epoch1.shape}",
            'matches': False
        }
    
    inc_diff = torch.abs(sim_inc_tensor - real_incentives_epoch1)
    inc_max_diff = torch.max(inc_diff).item()
    inc_mean_diff = torch.mean(inc_diff).item()
    inc_nonzero_diff = (inc_diff > tolerance).sum().item()
    
    return {
        'max_diff': inc_max_diff,
        'mean_diff': inc_mean_diff,
        'nonzero_diffs': inc_nonzero_diff,
        'matches': inc_max_diff < tolerance,
        'shape': list(sim_inc_tensor.shape)
    }


def validate_simulator(
    start_date: datetime,
    end_date: datetime,
    netuid: int = 1,
    tolerance: float = 1e-4,
    max_epochs: int = 3,
    generate_diagnostics: bool = True,
    bond_penalty_override: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Validate simulator against real metagraph data using functional approach.
    
    Args:
        start_date: Start date for data fetching
        end_date: End date for data fetching
        netuid: Network UID to test on
        tolerance: Tolerance for numerical comparisons
        max_epochs: Maximum number of epochs to validate (default: 3)
        generate_diagnostics: Whether to generate diagnostic artifacts on failure (default: True)
        
    Returns:
        Dict with validation results and statistics
    """
    logger.info(f"Starting validation for netuid {netuid} from {start_date} to {end_date} (max {max_epochs} epochs)")
    
    # 1. Prepare metagraph data
    case, tested_blocks = prepare_metagraph_data(start_date, end_date, netuid, max_epochs)
    
    # 2. Setup Yuma configuration
    yuma_config, is_yuma3_on = setup_yuma_configuration(netuid, bond_penalty_override)
    
    # 3. Run simulation
    yuma_simulation_name = YumaSimulationNames().YUMA3 if is_yuma3_on else YumaSimulationNames().YUMA2
    yuma_version = "YUMA3" if is_yuma3_on else "YUMA2"
    logger.info(f"Running {yuma_version} simulation...")
    sim_dividends, _, sim_bonds, sim_incentives_per_epoch, sim_normalized_dividends = _run_dynamic_simulation(
        case=case,
        yuma_version=yuma_simulation_name,
        yuma_config=yuma_config
    )
    
    # 4. Extract real data for comparison
    actual_epochs = case.num_epochs
    last_epoch_idx = actual_epochs - 1
    
    # Check if epoch 0 bonds were used as B_old
    has_epoch0_bonds = (hasattr(case, 'bonds_epochs') and 
                       len(case.bonds_epochs) > 0 and 
                       case.bonds_epochs[0] is not None)
    
    # If epoch 0 bonds were used as B_old:
    # - sim_bonds[0] corresponds to real epoch 1 (processed with epoch 0 bonds)
    # - sim_bonds[1] corresponds to real epoch 2, etc.
    # So we compare sim_bonds[last-1] with real epoch[last]
    if has_epoch0_bonds:
        # Simulation outputs are offset by 1
        sim_comparison_idx = last_epoch_idx - 1
        logger.info(f"Epoch 0 bonds used as B_old: comparing sim_bonds[{sim_comparison_idx}] with real epoch {last_epoch_idx}")
    else:
        # Direct correspondence
        sim_comparison_idx = last_epoch_idx
        logger.info(f"No epoch 0 bonds: comparing sim_bonds[{sim_comparison_idx}] with real epoch {last_epoch_idx}")
    
    # IMPORTANT: Use FULL bonds data (not filtered) to match simulation output
    real_bonds_last = case.metas[last_epoch_idx].get("bonds", None) if last_epoch_idx < len(case.metas) else None
    real_incentives_last = case.incentives_epochs[last_epoch_idx] if len(case.incentives_epochs) > last_epoch_idx else None
    real_dividends_last = case.dividends_epochs[last_epoch_idx] if len(case.dividends_epochs) > last_epoch_idx else None
    
    # 6. Initialize validation results
    validation_results = {
        'blocks_tested': tested_blocks,
        'epochs_tested': last_epoch_idx,  # We test the last epoch
        'epoch_0_bonds_used_as_B_old': has_epoch0_bonds,
        'simulation_epoch_offset': 1 if has_epoch0_bonds else 0,
        'comparing': f'sim_epoch_{sim_comparison_idx}_with_real_epoch_{last_epoch_idx}',
        'yuma_version': yuma_version,
        'comparisons': {},
        'summary': {
            'bonds_match': True,
            'dividends_match': True,
            'incentives_match': True,
            'total_differences': 0
        }
    }
    
    epoch_results = {}
    
    # 7. Compare bonds - extract validator rows from full matrix!
    bonds_result = compare_bonds(
        sim_bonds, 
        real_bonds_last,
        case.validators_epochs[last_epoch_idx] if len(case.validators_epochs) > last_epoch_idx else [],
        case.metas[last_epoch_idx]["hotkeys"] if last_epoch_idx < len(case.metas) else [],
        tolerance,
        case,
        sim_epoch_idx=sim_comparison_idx  # Pass the correct simulation epoch to compare
    )
    epoch_results['bonds'] = bonds_result
    
    if not bonds_result.get('matches', False):
        validation_results['summary']['bonds_match'] = False
        validation_results['summary']['total_differences'] += bonds_result.get('nonzero_diffs', 0)
    
    # 8. Compare dividends - extract correct epoch
    # Extract dividends for the correct simulation epoch
    sim_dividends_epoch = {}
    for validator_key, dividends_list in sim_normalized_dividends.items():
        if dividends_list and len(dividends_list) > sim_comparison_idx:
            sim_dividends_epoch[validator_key] = dividends_list[sim_comparison_idx]
        else:
            logger.warning(f"Missing dividend data for validator {validator_key} at epoch {sim_comparison_idx}")
    
    dividends_result = compare_dividends(
        sim_dividends_epoch,
        real_dividends_last,
        case.validators_epochs[last_epoch_idx] if len(case.validators_epochs) > last_epoch_idx else [],
        tolerance
    )
    epoch_results['dividends'] = dividends_result
    
    if not dividends_result.get('matches', False):
        validation_results['summary']['dividends_match'] = False
        validation_results['summary']['total_differences'] += dividends_result.get('nonzero_diffs', 0)
    
    # 9. Compare incentives - pass the full dict but specify which epoch to use
    incentives_result = compare_incentives(
        sim_incentives_per_epoch,
        real_incentives_last,
        case.servers[last_epoch_idx] if len(case.servers) > last_epoch_idx else [],
        tolerance,
        sim_epoch_idx=sim_comparison_idx
    )
    epoch_results['incentives'] = incentives_result
    
    if not incentives_result.get('matches', False):
        validation_results['summary']['incentives_match'] = False
        validation_results['summary']['total_differences'] += incentives_result.get('nonzero_diffs', 0)
    
    validation_results['comparisons'][f'epoch_{last_epoch_idx}'] = epoch_results
    
    # 10. Generate diagnostic artifacts if validation failed
    overall_success = all([
        validation_results['summary']['bonds_match'],
        validation_results['summary']['dividends_match'],
        validation_results['summary']['incentives_match']
    ])
    
    if not overall_success and generate_diagnostics:
        logger.info("Validation failed - generating diagnostic artifacts...")
        diagnostic_info = create_diagnostic_artifacts(
            netuid=netuid,
            validation_results=validation_results,
            case=case,
            sim_bonds=sim_bonds,
            sim_dividends=sim_normalized_dividends,
            sim_incentives=sim_incentives_per_epoch,
            tolerance=tolerance,
            yuma_config=yuma_config
        )
        validation_results['diagnostic_artifacts'] = diagnostic_info.get('artifact_path', None)
        logger.info(f"Diagnostic artifacts saved to: {diagnostic_info.get('artifact_path', 'N/A')}")
    
    return validation_results


def print_validation_results(results: Dict[str, Any]):
    """Print validation results in a readable format."""
    
    print(f"Blocks tested: {results['blocks_tested']}")
    print(f"Epochs tested: {results['epochs_tested']}")
    
    summary = results['summary']
    print("SUMMARY:")
    print(f"  Bonds match:     {'✓' if summary['bonds_match'] else '✗'}")
    print(f"  Dividends match: {'✓' if summary['dividends_match'] else '✗'}")
    print(f"  Incentives match:{'✓' if summary['incentives_match'] else '✗'}")
    print(f"  Total differences: {summary['total_differences']}")
    
    overall_success = all([
        summary['bonds_match'],
        summary['dividends_match'],
        summary['incentives_match']
    ])
    
    if overall_success:
        print("🎉 VALIDATION PASSED: Simulator matches real metagraph data!")
    else:
        print("❌ VALIDATION FAILED: Simulator differs from real metagraph data")
        
        # Print diagnostic artifact location if available
        if 'diagnostic_artifacts' in results and results['diagnostic_artifacts']:
            print(f"\n📁 Diagnostic artifacts saved to: {results['diagnostic_artifacts']}")
            print("   Check the diagnostic_report.json and summary.txt for detailed analysis")
        
        print("\nRESULTS OVERVIEW:")
        
        for epoch_name, epoch_data in results['comparisons'].items():
            print(f"\n{epoch_name.upper()}:")
            for metric, data in epoch_data.items():
                if 'error' in data:
                    print(f"  {metric}: ✗ ERROR - {data['error']}")
                else:
                    status = "✓" if data['matches'] else "✗"
                    extra_info = ""
                    if 'missing_validators' in data and data['missing_validators']:
                        extra_info += f", missing_validators: {data['missing_validators']}"
                    if 'missing_miners' in data and data['missing_miners']:
                        extra_info += f", missing_miners: {data['missing_miners']}"
                    
                    print(f"  {metric}: {status} (max_diff: {data.get('max_diff', 0):.6f}, mean_diff: {data.get('mean_diff', 0):.6f}, nonzero_diffs: {data.get('nonzero_diffs', 0)}{extra_info})")


def create_weight_analysis_report(
    case: 'MetagraphCase',
    target_miner_uid: int,
    netuid: int
) -> Dict[str, Any]:
    """
    Create a comprehensive weight analysis report for a specific miner UID across all epochs.
    This helps debug incentive divergences by showing exactly what weights validators are setting.
    
    Args:
        case: MetagraphCase with metagraph data
        target_miner_uid: The miner UID to analyze
        netuid: Network UID
        
    Returns:
        Dict containing detailed weight analysis across all epochs
    """
    
    report = {
        "metadata": {
            "target_miner_uid": target_miner_uid,
            "netuid": netuid,
            "blocks": [case.metas[i].get("block", f"epoch_{i}") for i in range(len(case.metas))],
            "analysis_purpose": "Debug incentive divergence by analyzing validator weights to target miner"
        },
        "epochs": []
    }
    
    # Process each epoch
    for epoch_idx in range(len(case.metas)):
        epoch_meta = case.metas[epoch_idx]
        
        # Get weight matrix for this epoch
        W_full = epoch_meta.get("W", None)
        if W_full is None or not isinstance(W_full, torch.Tensor):
            continue
            
        # Get hotkeys for this epoch
        hotkeys = epoch_meta.get("hotkeys", [])
        if target_miner_uid >= len(hotkeys):
            continue
            
        target_hotkey = hotkeys[target_miner_uid]
        if target_hotkey is None:
            logger.warning(f"Target miner UID {target_miner_uid} has None hotkey in epoch {epoch_idx}, skipping")
            continue
        
        # Get validator info for this epoch
        validator_uids = case.valid_indices_epochs[epoch_idx] if epoch_idx < len(case.valid_indices_epochs) else []
        
        epoch_data = {
            "epoch": epoch_idx,
            "block": epoch_meta.get("block", f"epoch_{epoch_idx}"),
            "target_miner_hotkey": target_hotkey[:20] + "..." if len(target_hotkey) > 20 else target_hotkey,
            "validators": {},
            "weight_statistics": {
                "total_weight_to_target": 0.0,
                "num_validators_setting_weight": 0,
                "max_weight": 0.0,
                "min_nonzero_weight": float('inf'),
                "weight_distribution": []
            }
        }
        
        total_weight = 0.0
        validators_with_weight = 0
        all_weights = []
        
        # Analyze each validator's weight to the target miner
        for validator_uid in validator_uids:
            if validator_uid >= W_full.shape[0]:
                continue
                
            # Get validator info
            validator_hotkey = hotkeys[validator_uid] if validator_uid < len(hotkeys) else f"UID{validator_uid}"
            validator_hotkey_short = validator_hotkey[:20] + "..." if len(validator_hotkey) > 20 else validator_hotkey
            
            # Get weight from validator to target miner
            weight_to_target = float(W_full[validator_uid, target_miner_uid])
            
            # Get validator's stake
            S = epoch_meta.get("S", torch.zeros(256))
            stake = float(S[validator_uid]) if validator_uid < S.shape[0] else 0.0
            
            # Get validator's total outgoing weights (row sum)
            total_outgoing = float(W_full[validator_uid, :].sum())
            
            # Store validator data
            epoch_data["validators"][f"uid_{validator_uid}"] = {
                "validator_uid": validator_uid,
                "validator_hotkey": validator_hotkey_short,
                "weight_to_target": weight_to_target,
                "weight_percentage_of_validator": (weight_to_target / total_outgoing * 100) if total_outgoing > 0 else 0.0,
                "validator_stake": stake,
                "validator_total_outgoing_weights": total_outgoing
            }
            
            # Update statistics
            total_weight += weight_to_target
            if weight_to_target > 0:
                validators_with_weight += 1
                all_weights.append(weight_to_target)
                epoch_data["weight_statistics"]["max_weight"] = max(epoch_data["weight_statistics"]["max_weight"], weight_to_target)
                if weight_to_target < epoch_data["weight_statistics"]["min_nonzero_weight"]:
                    epoch_data["weight_statistics"]["min_nonzero_weight"] = weight_to_target
        
        # Finalize statistics
        epoch_data["weight_statistics"]["total_weight_to_target"] = total_weight
        epoch_data["weight_statistics"]["num_validators_setting_weight"] = validators_with_weight
        epoch_data["weight_statistics"]["weight_distribution"] = sorted(all_weights, reverse=True)
        
        if epoch_data["weight_statistics"]["min_nonzero_weight"] == float('inf'):
            epoch_data["weight_statistics"]["min_nonzero_weight"] = 0.0
        
        # Add consensus weight calculation info
        if len(all_weights) > 0:
            epoch_data["weight_statistics"]["mean_weight"] = sum(all_weights) / len(all_weights)
            epoch_data["weight_statistics"]["median_weight"] = sorted(all_weights)[len(all_weights)//2] if all_weights else 0.0
        else:
            epoch_data["weight_statistics"]["mean_weight"] = 0.0
            epoch_data["weight_statistics"]["median_weight"] = 0.0
            
        report["epochs"].append(epoch_data)
    
    return report


def create_bond_evolution_report(
    case: 'MetagraphCase',
    sim_bonds: List[torch.Tensor],
    target_miner_uid: int,
    focus_validator_uid: int,
    netuid: int,
    tested_blocks: List[int]
) -> Dict[str, Any]:
    """
    Creates a comprehensive bond evolution report tracking how validator bonds to a specific miner
    evolve across epochs. Shows both simulated and real bond values for analysis.
    
    This function tracks bond evolution patterns by examining all validators' bonds toward 
    a target miner across multiple epochs, with special focus on one validator's behavior.
    Includes stake information and weight values to provide context for bond decisions.
    
    Args:
        case: MetagraphCase containing real metagraph data across epochs
        sim_bonds: List of simulated bond tensors, one per epoch
        target_miner_uid: UID of the miner to track bonds toward
        focus_validator_uid: UID of validator to focus analysis on
        netuid: Network ID being analyzed
        tested_blocks: List of block numbers corresponding to each epoch
        
    Returns:
        Dict containing comprehensive bond evolution data with blocks instead of timestamps
    """
    
    bond_report = {
        "metadata": {
            "target_miner_uid": target_miner_uid,
            "focus_validator_uid": focus_validator_uid,
            "netuid": netuid,
            "blocks": tested_blocks  # Use blocks instead of timestamp
        },
        "epochs": []
    }
    
    # Process each epoch
    for epoch_idx in range(len(case.metas)):
        if epoch_idx >= len(sim_bonds):
            break
            
        epoch_meta = case.metas[epoch_idx]
        sim_bonds_epoch = sim_bonds[epoch_idx]
        
        # Get real bonds for this epoch - both raw and normalized versions
        real_bonds_raw = epoch_meta.get("bonds", torch.zeros(1, 1))
        real_bonds_normalized = torch.zeros(1, 1)
        real_bonds_colsum = torch.zeros(1, 1)
        
        if isinstance(real_bonds_raw, torch.Tensor) and len(real_bonds_raw.shape) == 2:
            # Normalize from fixed point (divide by 65535)
            real_bonds_normalized = real_bonds_raw / 65535.0
            # Apply column-sum normalization to the normalized bonds (not raw)
            real_bonds_colsum = real_bonds_normalized / (real_bonds_normalized.sum(dim=0, keepdim=True) + 1e-8)
        
        # Extract active validator and miner UIDs for this epoch
        # Try to get from case structure first, fallback to epoch_meta
        if epoch_idx < len(case.valid_indices_epochs):
            validator_uids = case.valid_indices_epochs[epoch_idx]
        else:
            validator_uids = epoch_meta.get("validator_uids", [])
        
        if epoch_idx < len(case.miner_indices_epochs):
            miner_uids = case.miner_indices_epochs[epoch_idx]
        else:
            miner_uids = epoch_meta.get("miner_uids", [])
        
        hotkeys = epoch_meta.get("hotkeys", [])
        
        epoch_data = {
            "epoch": epoch_idx,
            "validators": {},
            "summary": {},
            "bonds_property_shape": list(sim_bonds_epoch.shape) if hasattr(sim_bonds_epoch, 'shape') else [],
            "bonds_property_sum": float(sim_bonds_epoch.sum()) if hasattr(sim_bonds_epoch, 'sum') else 0
        }
        
        # Track bond data for each validator toward target miner
        sim_bonds_to_target_sum = 0.0
        sim_bonds_to_target_nonzero = 0
        
        for val_idx, validator_uid in enumerate(validator_uids):
            if val_idx >= sim_bonds_epoch.shape[0]:
                break
                
            # Find target miner index in miner_uids list
            target_miner_idx = None
            try:
                if target_miner_uid in miner_uids:
                    target_miner_idx = miner_uids.index(target_miner_uid)
            except (ValueError, TypeError):
                pass
            
            # Extract bond values
            sim_bond_value = 0.0
            real_bond_raw = 0.0 
            real_bond_normalized = 0.0
            real_bond_colsum = 0.0
            
            if target_miner_idx is not None and target_miner_idx < sim_bonds_epoch.shape[1]:
                sim_bond_value = float(sim_bonds_epoch[val_idx, target_miner_idx])
                
                # Real bond extraction - use validator_uid not val_idx!
                if (validator_uid < real_bonds_raw.shape[0] and 
                    target_miner_uid < real_bonds_raw.shape[1]):
                    real_bond_raw = float(real_bonds_raw[validator_uid, target_miner_uid])
                    real_bond_normalized = real_bond_raw / 65535.0  # Fixed point normalization
                    # Successfully extracted real bond data
                
                if (validator_uid < real_bonds_colsum.shape[0] and 
                    target_miner_uid < real_bonds_colsum.shape[1]):
                    real_bond_colsum = float(real_bonds_colsum[validator_uid, target_miner_uid])
            
            # Update summary counters
            sim_bonds_to_target_sum += sim_bond_value
            if sim_bond_value > 0:
                sim_bonds_to_target_nonzero += 1
            
            # Get validator's weight toward target miner
            weight_value = 0.0
            if "W" in epoch_meta and isinstance(epoch_meta["W"], torch.Tensor):
                W = epoch_meta["W"]
                if (validator_uid < W.shape[0] and target_miner_uid < W.shape[1]):
                    weight_value = float(W[validator_uid, target_miner_uid])
            
            # Get validator's stake
            stake_value = 0.0
            if "S" in epoch_meta and isinstance(epoch_meta["S"], torch.Tensor):
                S = epoch_meta["S"]
                if validator_uid < S.shape[0]:
                    stake_value = float(S[validator_uid])
            
            # Get validator hotkey for identification
            hotkeys = epoch_meta.get("hotkeys", [])
            validator_hotkey = "N/A"
            if validator_uid < len(hotkeys):
                validator_hotkey = hotkeys[validator_uid][:16] + "..." if len(hotkeys[validator_uid]) > 16 else hotkeys[validator_uid]
            
            # Store validator data
            epoch_data["validators"][f"uid_{validator_uid}"] = {
                "validator_uid": validator_uid,
                "validator_hotkey": validator_hotkey,
                "bonds": {
                    "simulated": sim_bond_value,
                    "real_raw": real_bond_raw,
                    "real_normalized": real_bond_normalized,
                    "real_col_sum": real_bond_colsum
                },
                "weights": {
                    "value": weight_value
                },
                "stake": stake_value
            }
        
        # Add summary data
        epoch_data["summary"] = {
            "sim_bonds_to_target_sum": sim_bonds_to_target_sum,
            "sim_bonds_to_target_nonzero": sim_bonds_to_target_nonzero,
            "num_validators": len(validator_uids)
        }
        
        bond_report["epochs"].append(epoch_data)
    
    return bond_report


def main():
    """Main validation function."""
    import argparse
    parser = argparse.ArgumentParser(description='Validate Yuma simulator against real metagraph data')
    
    # Support both single subnet, multiple subnets, and top subnets
    subnet_group = parser.add_mutually_exclusive_group(required=True)
    subnet_group.add_argument('--netuid', type=int, help='Single subnet ID to validate')
    subnet_group.add_argument('--netuids', type=str, help='Comma-separated list of subnet IDs to validate (e.g., "1,3,9")')
    subnet_group.add_argument('--top-subnets', type=int, metavar='N', help='Validate top N subnets by TAO emission')
    
    parser.add_argument('--hours', type=float, default=None, help='Number of hours of data to fetch (default: auto-calculated based on max-epochs)')
    parser.add_argument('--use-epoch-time', action='store_true', help='Calculate time range based on max-epochs * 72 minutes per epoch')
    parser.add_argument('--days-ago', type=int, default=1, help='Days ago to end data fetch (default: 1)')
    parser.add_argument('--tolerance', type=float, default=1e-4, help='Numerical tolerance for comparisons (default: 1e-4)')
    parser.add_argument('--max-epochs', type=int, default=3, help='Maximum number of epochs to validate (default: 3)')
    parser.add_argument('--no-diagnostics', action='store_true', help='Disable diagnostic artifact generation on failure')
    parser.add_argument('--bond-penalty-override', type=float, default=None, help='Override bond_penalty (e.g., 1.0) for parity testing')
    args = parser.parse_args()
    
    # Parse subnet IDs
    if args.netuid:
        subnet_ids = [args.netuid]
    elif args.top_subnets:
        logger.info(f"Fetching top {args.top_subnets} subnets by TAO emission...")
        subnet_ids = get_top_subnets_by_tao_emission(args.top_subnets)
        if not subnet_ids:
            logger.error("Failed to fetch top subnets from the chain")
            return 1
        logger.info(f"Selected top subnets: {subnet_ids}")
    else:
        try:
            subnet_ids = [int(id.strip()) for id in args.netuids.split(',')]
        except ValueError:
            logger.error(f"Invalid netuids format: {args.netuids}. Expected comma-separated integers.")
            return 1
    
    # Calculate date range
    end_date = datetime.now() - timedelta(days=args.days_ago)
    
    # Auto-calculate time range based on epochs (unless --hours explicitly provided)
    epoch_duration_minutes = 72
    if args.hours is None:
        # Auto-calculate: max_epochs * 72 minutes + 20% buffer for safety
        total_minutes = int(args.max_epochs * epoch_duration_minutes * 1.2)
        start_date = end_date - timedelta(minutes=total_minutes)
        logger.info(f"Auto-calculated time range: {args.max_epochs} epochs * {epoch_duration_minutes} min * 1.2 = {total_minutes} min")
    elif args.use_epoch_time:
        # Use exact epoch-based calculation
        total_minutes = args.max_epochs * epoch_duration_minutes
        start_date = end_date - timedelta(minutes=total_minutes)
        logger.info(f"Using epoch-based time calculation: {args.max_epochs} epochs * {epoch_duration_minutes} min = {total_minutes} min")
    else:
        # Use explicit hours
        start_date = end_date - timedelta(hours=args.hours)
        logger.info(f"Using explicit time range: {args.hours} hours")
    
    logger.info(f"Validating subnets {subnet_ids} from {start_date} to {end_date}")
    
    try:
        # Fetch hyperparameters for all subnets first for comparison
        if len(subnet_ids) > 1:
            logger.info(f"\n{'='*60}")
            logger.info(f"HYPERPARAMETERS COMPARISON FOR SUBNETS {subnet_ids}")
            logger.info(f"{'='*60}")
            
            all_hyperparams = fetch_multiple_subnets_hyperparameters(subnet_ids)
            
            # Show Yuma versions first
            logger.info(f"\nYuma Versions:")
            for netuid in subnet_ids:
                if netuid in all_hyperparams:
                    yuma_version = "YUMA3" if all_hyperparams[netuid].get('is_yuma3_on', False) else "YUMA2"
                    logger.info(f"  Subnet {netuid}: {yuma_version}")
                else:
                    logger.info(f"  Subnet {netuid}: No data available")
            
            # Compare key hyperparameters
            key_params = ['bonds_moving_avg', 'liquid_alpha_enabled', 'commit_reveal_weights_enabled', 
                         'alpha_low', 'alpha_high', 'kappa']
            
            for param in key_params:
                logger.info(f"\n{param}:")
                for netuid in subnet_ids:
                    if netuid in all_hyperparams and 'hyperparams' in all_hyperparams[netuid]:
                        value = all_hyperparams[netuid]['hyperparams'].get(param, 'N/A')
                        logger.info(f"  Subnet {netuid}: {value}")
                    else:
                        logger.info(f"  Subnet {netuid}: No data available")
        
        # Validate each subnet
        all_results = {}
        all_success = True
        
        for netuid in subnet_ids:
            logger.info(f"\n{'='*60}")
            logger.info(f"VALIDATING SUBNET {netuid}")
            logger.info(f"{'='*60}")
            
            try:
                subnet_results = validate_simulator(
                    start_date=start_date,
                    end_date=end_date,
                    netuid=netuid,
                    tolerance=args.tolerance,
                    max_epochs=args.max_epochs,
                    generate_diagnostics=not args.no_diagnostics,
                    bond_penalty_override=args.bond_penalty_override
                )
                
                all_results[netuid] = subnet_results
                
                # Check if this subnet passed validation
                subnet_success = all([
                    subnet_results['summary']['bonds_match'],
                    subnet_results['summary']['dividends_match'],
                    subnet_results['summary']['incentives_match']
                ])
                
                if not subnet_success:
                    all_success = False
                    
            except Exception as e:
                logger.error(f"Validation failed for subnet {netuid}: {e}")
                import traceback
                traceback.print_exc()
                all_results[netuid] = {'error': str(e), 'success': False}
                all_success = False
        
        # Print consolidated results
        print("\n" + "="*80)
        print("CONSOLIDATED VALIDATION RESULTS")
        print("="*80)
        
        for netuid, results in all_results.items():
            print(f"\nSUBNET {netuid}:")
            if 'error' in results:
                print(f"  ❌ FAILED: {results['error']}")
            else:
                print_validation_results(results)
        
        # Overall summary
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY:")
        successful_subnets = [netuid for netuid, results in all_results.items() if 'error' not in results and all([
            results['summary']['bonds_match'],
            results['summary']['dividends_match'], 
            results['summary']['incentives_match']
        ])]
        failed_subnets = [netuid for netuid in subnet_ids if netuid not in successful_subnets]
        
        print(f"Successful subnets: {successful_subnets}")
        print(f"Failed subnets: {failed_subnets}")
        print(f"Overall success: {'✓' if all_success else '✗'}")
        
        sys.exit(0 if all_success else 1)
        
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == '__main__':
    main()
