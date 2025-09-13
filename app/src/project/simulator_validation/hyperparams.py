import os
import logging
from typing import Dict, Any, Optional, List, Tuple

import requests

from project.yuma_simulation._internal.yumas import (
    YumaConfig,
    YumaParams,
    SimulationHyperparameters,
)

logger = logging.getLogger(__name__)


def normalize_hyperparameters(raw_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize raw hyperparameters from blockchain/service format to simulator format.
    Raises KeyError if required params are missing.
    """
    required = [
        "alpha_low",
        "alpha_high",
        "kappa",
        "bonds_moving_avg",
        "liquid_alpha_enabled",
    ]
    missing = [p for p in required if p not in raw_params]
    if missing:
        available = list(raw_params.keys())
        raise KeyError(
            f"Missing required hyperparameters: {missing}\nAvailable parameters: {available}"
        )

    normalized = {
        "alpha_low": raw_params["alpha_low"] / 65535.0,
        "alpha_high": raw_params["alpha_high"] / 65535.0,
        "kappa": raw_params["kappa"] / 65535.0,
        "bonds_moving_avg": raw_params["bonds_moving_avg"] / 1_000_000.0,
        "liquid_alpha_enabled": bool(raw_params["liquid_alpha_enabled"]),
    }

    # Optional parameters
    normalized["alpha_sigmoid_steepness"] = float(
        raw_params.get("alpha_sigmoid_steepness", 10.0)
    )

    # bond_penalty: prefer explicit field; fallback to 1.0 if missing
    bond_penalty = None
    if "bond_penalty" in raw_params:
        bp_raw = raw_params["bond_penalty"]
        try:
            bp_val = float(bp_raw)
            if bp_val > 1.0:
                bp_val = bp_val / 65535.0
            bond_penalty = max(0.0, min(1.0, bp_val))
            logger.info(
                f"Using bond_penalty from hyperparams: raw={bp_raw}, normalized={bond_penalty}"
            )
        except Exception:
            logger.warning(f"Failed to parse bond_penalty from hyperparams: {bp_raw}")
    if bond_penalty is None:
        bond_penalty = 1.0
    normalized["bond_penalty"] = bond_penalty

    # Optional: commit–reveal controls (names vary by source)
    try:
        cr_enabled = raw_params.get("commit_reveal_weights_enabled")
        if cr_enabled is None:
            cr_enabled = raw_params.get("commit_reveal_enabled")
        normalized["commit_reveal_enabled"] = bool(cr_enabled) if cr_enabled is not None else False
    except Exception:
        normalized["commit_reveal_enabled"] = False

    try:
        # Try common keys for the period (epochs)
        period = (
            raw_params.get("reveal_period_epochs")
            or raw_params.get("commit_reveal_period")
            or raw_params.get("commit_reveal_interval")
        )
        normalized["commit_reveal_period_epochs"] = int(period) if period is not None else 0
    except Exception:
        normalized["commit_reveal_period_epochs"] = 0

    return normalized


def get_top_subnets_by_tao_emission(top_n: int = 10) -> List[int]:
    """Return netuids of top-N subnets by `tao_in_emission` (excludes netuid 0)."""
    try:
        import bittensor as bt

        s = bt.subtensor()
        all_subnets = list(s.all_subnets())
        all_subnets.sort(key=lambda sn: sn.tao_in_emission, reverse=True)
        filtered = [sn for sn in all_subnets if sn.netuid != 0]
        return [sn.netuid for sn in filtered[:top_n]]
    except Exception as e:
        logger.error(f"Error fetching top subnets: {e}")
        return []


def fetch_subnet_hyperparameters(
    netuid: int, service_url: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Fetch hyperparameters for a specific subnet from the hyperparameters service."""
    if service_url is None:
        service_url = os.getenv("HYPERPARAMS_SERVICE_URL", "http://localhost:8000/")
    if not service_url.endswith("/"):
        service_url += "/"

    url = f"{service_url}hyperparams/subnet/{netuid}/"
    try:
        logger.info(f"Fetching hyperparameters from {url}")
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data
        if response.status_code == 404:
            logger.warning(f"No hyperparameters found for subnet {netuid}")
            return None
        logger.error(f"Failed to fetch hyperparameters: HTTP {response.status_code}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching hyperparameters: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching hyperparameters: {e}")
        return None


def fetch_multiple_subnets_hyperparameters(
    subnet_ids: List[int], service_url: Optional[str] = None
) -> Dict[int, Dict[str, Any]]:
    """Fetch hyperparameters for multiple subnets using the batch endpoint."""
    if service_url is None:
        service_url = os.getenv("HYPERPARAMS_SERVICE_URL", "http://localhost:8000/")
    if not service_url.endswith("/"):
        service_url += "/"

    subnet_ids_str = ",".join(map(str, subnet_ids))
    url = f"{service_url}hyperparams/multiple/?subnet_ids={subnet_ids_str}"
    try:
        logger.info(f"Fetching hyperparameters for subnets {subnet_ids} from {url}")
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            result: Dict[int, Dict[str, Any]] = {}
            for subnet_data in data.get("subnets", []):
                subnet_id = subnet_data["subnet"]
                result[subnet_id] = subnet_data
            not_found = data.get("not_found", [])
            if not_found:
                logger.warning(f"Hyperparameters not found for subnets: {not_found}")
            return result
        if response.status_code == 404:
            logger.warning(
                f"No hyperparameters found for any of the subnets {subnet_ids}"
            )
            return {}
        logger.error(f"Failed to fetch hyperparameters: HTTP {response.status_code}")
        return {}
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching hyperparameters for subnets {subnet_ids}: {e}")
        return {}
    except Exception as e:
        logger.error(
            f"Unexpected error fetching hyperparameters for subnets {subnet_ids}: {e}"
        )
        return {}


def setup_yuma_configuration(
    netuid: int,
    bond_penalty_override: Optional[float] = None,
    hyperparams_data: Optional[Dict[str, Any]] = None,
) -> Tuple[YumaConfig, bool, Dict[str, Any]]:
    """
    Fetch subnet hyperparams and build YumaConfig; returns (config, is_yuma3_on).
    Raises ValueError when hyperparams are missing or invalid.
    """
    from .hyperparams import fetch_subnet_hyperparameters  # local import for clarity

    # Allow caller to inject already-fetched hyperparams (e.g., from multi-subnet endpoint)
    if hyperparams_data is None:
        hyperparams_data = fetch_subnet_hyperparameters(netuid)
    else:
        logger.info(f"Using injected hyperparameters for subnet {netuid} (skipping HTTP fetch)")
    if not hyperparams_data or "hyperparams" not in hyperparams_data:
        raise ValueError(
            f"Failed to fetch hyperparameters for subnet {netuid}. Cannot proceed with validation."
        )

    raw_hyperparams = hyperparams_data["hyperparams"]
    is_yuma3_on = hyperparams_data.get("is_yuma3_on", False)

    try:
        params = normalize_hyperparameters(raw_hyperparams)
        effective_bond_penalty = params["bond_penalty"]
        if bond_penalty_override is not None:
            logger.info(
                f"Overriding bond_penalty to {bond_penalty_override} for parity/testing"
            )
            effective_bond_penalty = float(bond_penalty_override)

        yuma_config = YumaConfig(
            simulation=SimulationHyperparameters(
                kappa=params["kappa"],
                bond_penalty=effective_bond_penalty,
                total_epoch_emission=100.0,
                total_subnet_stake=1_000_000.0,
            ),
            yuma_params=YumaParams(
                bond_moving_avg=params["bonds_moving_avg"],
                liquid_alpha=params["liquid_alpha_enabled"],
                alpha_high=params["alpha_high"],
                alpha_low=params["alpha_low"],
                alpha_sigmoid_steepness=params["alpha_sigmoid_steepness"],
            ),
        )
        cr_info = {
            "commit_reveal_enabled": params.get("commit_reveal_enabled", False),
            "commit_reveal_period_epochs": params.get("commit_reveal_period_epochs", 0),
        }
        return yuma_config, is_yuma3_on, cr_info
    except KeyError as e:
        logger.error(f"Hyperparameter validation failed: {e}")
        raise ValueError(f"Cannot proceed with validation: {e}")
    except Exception as e:
        logger.error(f"Error processing hyperparameters: {e}")
        raise
