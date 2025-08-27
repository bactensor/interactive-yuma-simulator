import logging
from typing import Dict, Any, List, Optional

import torch

logger = logging.getLogger(__name__)


def compare_bonds(
    sim_bonds: List[torch.Tensor],
    real_bonds_full: torch.Tensor,
    validators_epoch: List[str],
    epoch_hotkeys: List[str],
    tolerance: float,
    sim_epoch_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """Compare simulation bonds with real bonds (filtered to validator rows)."""
    if len(sim_bonds) == 0 or real_bonds_full is None:
        return {"error": "Missing bonds data for comparison", "matches": False}

    # Choose simulation epoch
    if sim_epoch_idx is not None:
        if sim_epoch_idx < 0 or sim_epoch_idx >= len(sim_bonds):
            return {
                "error": f"Invalid sim_epoch_idx {sim_epoch_idx} for {len(sim_bonds)} simulation epochs",
                "matches": False,
            }
        sim_bonds_epoch = sim_bonds[sim_epoch_idx]
    else:
        sim_bonds_epoch = sim_bonds[-1]

    # Normalize real bonds to [0,1] and column-normalize
    real_bonds_normalized = real_bonds_full / 65535.0
    col_sums = real_bonds_normalized.sum(dim=0, keepdim=True)
    col_sums = torch.where(col_sums > 1e-6, col_sums, torch.ones_like(col_sums))
    real_bonds_normalized = real_bonds_normalized / col_sums

    # Map validator hotkeys to uids (rows in full 256x256)
    validator_positions = []
    for hk in validators_epoch:
        try:
            validator_positions.append(epoch_hotkeys.index(hk))
        except ValueError:
            validator_positions.append(None)

    # Build filtered real bonds with rows in the same order as sim_bonds_epoch
    rows = []
    for pos in validator_positions:
        if pos is None:
            rows.append(torch.zeros_like(real_bonds_normalized[0]))
        else:
            rows.append(real_bonds_normalized[pos])
    real_bonds_filtered = torch.stack(rows) if rows else torch.zeros_like(sim_bonds_epoch)

    # Sanity shape check
    if sim_bonds_epoch.shape != real_bonds_filtered.shape:
        return {
            "error": f"Shape mismatch: sim={tuple(sim_bonds_epoch.shape)}, real={tuple(real_bonds_filtered.shape)}",
            "matches": False,
        }

    bonds_diff = torch.abs(sim_bonds_epoch - real_bonds_filtered)
    bonds_max_diff = float(torch.max(bonds_diff).item())
    bonds_mean_diff = float(torch.mean(bonds_diff).item())
    bonds_nonzero_diff = int((bonds_diff > tolerance).sum().item())

    # Non-zero stats
    sim_nonzero_count = int((sim_bonds_epoch > 0).sum().item())
    real_nonzero_count = int((real_bonds_filtered > 0).sum().item())

    # Log top differing positions for debugging
    any_nonzero_mask = (sim_bonds_epoch > 0) | (real_bonds_filtered > 0)
    if any_nonzero_mask.any():
        diffs = []
        idxs = torch.nonzero(any_nonzero_mask, as_tuple=False)
        for row, col in idxs.tolist():
            sim_val = float(sim_bonds_epoch[row, col].item())
            real_val = float(real_bonds_filtered[row, col].item())
            diff = abs(sim_val - real_val)
            diffs.append((diff, row, col, sim_val, real_val))
        diffs.sort(key=lambda x: x[0], reverse=True)
        for i, (diff, row, col, sim_val, real_val) in enumerate(diffs[:10]):
            logger.info(
                f"  Position [{row},{col}]: Sim={sim_val:.6f}, Real={real_val:.6f}, Diff={diff:.6f}"
            )
    else:
        logger.info("No non-zero bonds found in either simulation or real data")

    return {
        "max_diff": bonds_max_diff,
        "mean_diff": bonds_mean_diff,
        "nonzero_diffs": bonds_nonzero_diff,
        "matches": bonds_max_diff < tolerance,
        "sim_shape": list(sim_bonds_epoch.shape),
        "real_shape": list(real_bonds_filtered.shape),
        "sim_nonzero_bonds": sim_nonzero_count,
        "real_nonzero_bonds": real_nonzero_count,
        "note": "Direct comparison of filtered bond matrices",
    }


def compare_dividends(
    sim_normalized_dividends: Dict,
    real_dividends_epoch: torch.Tensor,
    validators_epoch: List[str],
    tolerance: float,
) -> Dict[str, Any]:
    """Compare simulation dividends with real dividends for a single epoch."""
    if real_dividends_epoch is None:
        return {"error": "No real dividends data available", "matches": False}

    sim_div_values: List[float] = []
    for hk in validators_epoch:
        if hk in sim_normalized_dividends:
            data = sim_normalized_dividends[hk]
            if isinstance(data, list) and data:
                sim_div_values.append(float(data[-1]))
            elif isinstance(data, (int, float)):
                sim_div_values.append(float(data))
            else:
                sim_div_values.append(0.0)
        else:
            sim_div_values.append(0.0)

    if len(sim_div_values) != len(validators_epoch):
        return {
            "error": f"Simulation dividends length mismatch: got {len(sim_div_values)}, expected {len(validators_epoch)}",
            "matches": False,
        }

    sim_div_tensor = torch.tensor(sim_div_values, dtype=torch.float32)
    if sim_div_tensor.shape != real_dividends_epoch.shape:
        return {
            "error": f"Shape mismatch: sim={tuple(sim_div_tensor.shape)}, real={tuple(real_dividends_epoch.shape)}",
            "matches": False,
        }

    sim_scale = float(sim_div_tensor.max().item())
    real_scale = float(real_dividends_epoch.max().item())
    if sim_scale < 1e-6 and real_scale > 0.01:
        sim_div_tensor = sim_div_tensor * (real_scale / sim_scale if sim_scale > 0 else 0.0)
    elif real_scale > 10:
        real_dividends_epoch = real_dividends_epoch / 65535.0

    diff = torch.abs(sim_div_tensor - real_dividends_epoch)
    return {
        "max_diff": float(torch.max(diff).item()),
        "mean_diff": float(torch.mean(diff).item()),
        "nonzero_diffs": int((diff > tolerance).sum().item()),
        "matches": bool(torch.max(diff).item() < tolerance),
        "sim_nonzero_count": int((sim_div_tensor > 0).sum().item()),
        "real_nonzero_count": int((real_dividends_epoch > 0).sum().item()),
        "note": "Direct comparison using filtered dividend data",
    }


def compare_incentives(
    sim_incentives_per_epoch: Dict,
    real_incentives_epoch: torch.Tensor,
    miners: List[str],
    tolerance: float,
    sim_epoch_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """Compare simulation incentives with real incentives for a single epoch."""
    if real_incentives_epoch is None:
        return {"error": "No real incentives data available", "matches": False}

    sim_vals: List[float] = []
    for miner_hk in miners:
        if miner_hk in sim_incentives_per_epoch and sim_incentives_per_epoch[miner_hk]:
            seq = sim_incentives_per_epoch[miner_hk]
            if sim_epoch_idx is not None and 0 <= sim_epoch_idx < len(seq):
                sim_vals.append(float(seq[sim_epoch_idx]))
            else:
                sim_vals.append(float(seq[-1]))
        else:
            sim_vals.append(0.0)

    sim_tensor = torch.tensor(sim_vals, dtype=torch.float32)
    if sim_tensor.shape != real_incentives_epoch.shape:
        return {
            "error": f"Shape mismatch: sim={tuple(sim_tensor.shape)}, real={tuple(real_incentives_epoch.shape)}",
            "matches": False,
        }

    diff = torch.abs(sim_tensor - real_incentives_epoch)
    return {
        "max_diff": float(torch.max(diff).item()),
        "mean_diff": float(torch.mean(diff).item()),
        "nonzero_diffs": int((diff > tolerance).sum().item()),
        "matches": bool(torch.max(diff).item() < tolerance),
        "shape": list(sim_tensor.shape),
    }

