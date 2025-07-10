import math
import json
import textwrap
import uuid

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils

import torch

from project.yuma_simulation._internal.cases import BaseCase, MetagraphCase


def _get_validator_styles(
    validators: list[str],
) -> dict[str, tuple[str, str, int, int]]:
    combined_styles = [("-", "+", 12, 2), ("--", "x", 12, 1), (":", "o", 4, 1)]
    return {
        validator: combined_styles[idx % len(combined_styles)]
        for idx, validator in enumerate(validators)
    }


def _compute_mean(dividends: np.ndarray) -> float:
    """Computes the mean over valid epochs where the validator is present."""
    if np.all(np.isnan(dividends)):
        return 0.0
    return np.nanmean(dividends)


def _prepare_relative_dividends_data(
    validators_relative_dividends: dict[str, list[float]],
    case: BaseCase,
    num_epochs: int,
    epochs_padding: int = 0,
) -> dict | None:
    """
    Prepare common data for relative dividends plotting.

    Returns:
        dict with keys: plot_epochs, all_validators, top_vals, x, validator_styles, series_data
        or None if nothing to plot
    """
    plot_epochs = num_epochs - epochs_padding
    if plot_epochs <= 0 or not validators_relative_dividends:
        return None

    all_validators = list(validators_relative_dividends.keys())
    top_vals = getattr(case, "requested_validators", []) or all_validators.copy()
    if case.base_validator not in top_vals:
        top_vals.append(case.base_validator)

    x = list(range(plot_epochs))  # Using list for compatibility with both engines
    validator_styles = _get_validator_styles(all_validators)

    # Prepare series data
    series_data = []
    for idx, validator in enumerate(top_vals):
        series = validators_relative_dividends.get(validator, [])
        if len(series) <= epochs_padding:
            continue

        arr = np.array([d if d is not None else np.nan for d in series[epochs_padding:]], dtype=float)
        x_shifted = [x_val + idx * 0.05 for x_val in x]
        mean_pct = _compute_mean(arr) * 100
        label = f"{case.hotkey_label_map.get(validator, validator)}: total = {mean_pct:+.5f}%"
        ls, mk, ms, mew = validator_styles.get(validator, ("-", "o", 5, 1))

        series_data.append({
            'validator': validator,
            'data': arr,
            'x_shifted': x_shifted,
            'mean_pct': mean_pct,
            'label': label,
            'line_style': ls,
            'marker': mk,
            'marker_size': ms,
            'marker_edge_width': mew,
            'idx': idx
        })

    return {
        'plot_epochs': plot_epochs,
        'all_validators': all_validators,
        'top_vals': top_vals,
        'x': x,
        'validator_styles': validator_styles,
        'series_data': series_data
    }

def _get_relative_dividends_description_and_formula():
    """Get the description and formula for relative dividends."""
    description = (
        "'Validator Relative Dividends' is a performance metric that measures the deviation "
        "between a validator's actual dividends and the hypothetical zero-sum dividends they "
        "would have earned purely in proportion to their stake. This difference highlights "
        "whether a validator has over- or under-performed relative to the stake-weighted "
        "zero-sum baseline across the entire network."
    )

    formula_latex = (
        r"$\dfrac{\text{Validator's Dividends}}{\sum_{\text{all}}\text{Dividends}}"
        r" \;-\; "
        r"\dfrac{\text{Validator's Stake}}{\sum_{\text{all}}\text{Stake}}$"
    )

    formula_text = "(Validator's Dividends / Σ Dividends) - (Validator's Stake / Σ Stake)"

    return description, formula_latex, formula_text


def _prepare_bond_data_dynamic(
    bonds_per_epoch:    list[torch.Tensor],
    validators_epochs:  list[list[str]],
    miners_epochs:      list[list[str]],
    normalize:          bool = False,
) -> list[list[list[float]]]:
    """Optimized version using numpy operations and batch tensor conversion."""

    num_epochs = len(bonds_per_epoch)

    # Get unique validators while preserving order
    validator_keys = []
    seen = set()
    for vlist in validators_epochs:
        for v in vlist:
            if v not in seen:
                validator_keys.append(v)
                seen.add(v)

    max_miners = max(len(m) for m in miners_epochs)
    num_validators = len(validator_keys)

    # Pre-allocate result as numpy array for faster operations
    result_array = np.zeros((max_miners, num_validators, num_epochs), dtype=np.float32)

    # Pre-create validator key to global index mapping
    global_vmap = {v: i for i, v in enumerate(validator_keys)}

    for e in range(num_epochs):
        W = bonds_per_epoch[e]
        vkeys = validators_epochs[e]
        mkeys = miners_epochs[e]

        # Create index mappings
        local_to_global_v = [global_vmap[v] for v in vkeys]

        # Convert entire tensor to numpy once
        W_np = W.cpu().numpy()

        # Vectorized assignment
        for mi, miner in enumerate(mkeys):
            # Extract the entire row for this miner
            miner_weights = W_np[:, mi]  # All validators for this miner

            # Assign to global positions
            for local_vi, global_vi in enumerate(local_to_global_v):
                result_array[mi, global_vi, e] = miner_weights[local_vi]

    if normalize:
        # Vectorized normalization across epochs
        for t in range(num_epochs):
            epoch_slice = result_array[:, :, t]  # [miners, validators]
            totals = epoch_slice.sum(axis=1, keepdims=True)  # Sum per miner
            # Avoid division by zero
            mask = totals > 1e-12
            epoch_slice[mask.squeeze()] /= totals[mask.squeeze()]

    # Convert back to nested lists
    return result_array.tolist()


def _prepare_bonds_metagraph_data(
    case: MetagraphCase,
    bonds_per_epoch: list[torch.Tensor],
    default_miners: list[str],
    normalize: bool = False,
    epochs_padding: int = 0,
) -> dict | None:
    """
    Prepare common data for bonds metagraph plotting.

    Returns:
        dict with plotting data or None if nothing to plot
    """
    num_epochs = case.num_epochs
    plot_epochs = num_epochs - epochs_padding
    if plot_epochs <= 0:
        return None

    validators_epochs = case.validators_epochs
    miners_epochs = case.servers
    selected_validators = case.requested_validators

    bonds_data = _prepare_bond_data_dynamic(
        bonds_per_epoch, validators_epochs, miners_epochs,
        normalize=normalize,
    )

    subset_v = selected_validators or validators_epochs[0]
    subset_m = case.selected_servers or default_miners

    # Build validator keys list
    validator_keys: list[str] = []
    for epoch in validators_epochs:
        for v in epoch:
            if v not in validator_keys:
                validator_keys.append(v)

    m_idx = _find_indices(subset_m, miners_epochs, name="miner")
    v_idx = _find_indices(subset_v, validators_epochs, name="validator")

    # Prepare plot data
    plot_data: list[list[list[float]]] = []
    for mi in m_idx:
        per_val = []
        for vi in v_idx:
            series = []
            for e in range(epochs_padding, num_epochs):
                series.append(bonds_data[mi][vi][e])
            per_val.append(series)
        plot_data.append(per_val)

    return {
        'num_epochs': num_epochs,
        'plot_epochs': plot_epochs,
        'validators_epochs': validators_epochs,
        'miners_epochs': miners_epochs,
        'selected_validators': selected_validators,
        'subset_v': subset_v,
        'subset_m': subset_m,
        'validator_keys': validator_keys,
        'm_idx': m_idx,
        'v_idx': v_idx,
        'plot_data': plot_data,
        'x': list(range(plot_epochs))
    }


def _get_bonds_description_and_labels(normalize: bool) -> tuple[str, str, str]:
    """Get description, ylabel, and title suffix based on normalize flag."""
    if normalize:
        description = (
            "This plot shows each miner's normalized bond ratio from each validator over time. "
            "At every epoch, each miner's incoming bonds have been scaled so that their total across "
            "all validators equals 1."
        )
        ylabel = "Bond Ratio"
        title_suffix = " (Normalized)"
    else:
        description = (
            "This plot shows each validator's absolute bond value to each miner over time. "
            "At every epoch, the raw bond tensor is used, in the native units of a given simulation version."
        )
        ylabel = "Bond Value"
        title_suffix = ""

    return description, ylabel, title_suffix


from typing import Any, Sequence

def _find_indices(
    subset: Sequence[Any],
    epochs_list: Sequence[Sequence[Any]],
    name: str = "item",
) -> list[int]:
    """
    For each element in `subset`, find its .index(…) in the first
    sub‐sequence of `epochs_list` that contains it. If any element
    is missing, raise a RuntimeError with a clear message.
    """
    indices: list[int] = []
    for x in subset:
        # generator: e.index(x) only if x in e
        idx = next(
            (epoch.index(x) for epoch in epochs_list if x in epoch),
            None
        )
        if idx is None:
            raise RuntimeError(f"{name!r} {x!r} not found in any epoch")
        indices.append(idx)
    return indices


def _prepare_validator_server_weights_subplots_dynamic_data(
    case: MetagraphCase,
    default_miners: list[str],
    epochs_padding: int = 0,
) -> dict | None:
    """
    Prepare common data for validator server weights plotting.

    Returns:
        dict with plotting data or None if nothing to plot
    """
    total_epochs = case.num_epochs
    plot_epochs = total_epochs - epochs_padding
    if plot_epochs <= 0:
        return None

    subset_vals = case.requested_validators or case.validators_epochs[0]
    subset_srvs = case.selected_servers or default_miners
    hotkey_map = case.hotkey_label_map

    weights_epochs = case.weights_epochs
    validators_epochs = case.validators_epochs
    servers_epochs = case.servers

    # Build data cube
    data_cube: list[list[list[float]]] = []
    for srv in subset_srvs:
        per_val = []
        for val in subset_vals:
            series = []
            for e in range(epochs_padding, total_epochs):
                ve, se, W = validators_epochs[e], servers_epochs[e], weights_epochs[e]
                if (val in ve) and (srv in se):
                    r, c = ve.index(val), se.index(srv)
                    series.append(float(W[r, c].item()))
                else:
                    series.append(np.nan)
            per_val.append(series)
        data_cube.append(per_val)

    return {
        'total_epochs': total_epochs,
        'plot_epochs': plot_epochs,
        'subset_vals': subset_vals,
        'subset_srvs': subset_srvs,
        'hotkey_map': hotkey_map,
        'data_cube': data_cube,
        'x': list(range(plot_epochs))
    }


def _get_validator_weights_description() -> str:
    """Get the description for validator weights visualization."""
    return (
        "'Validators Weights per Miner' is a visualization that shows how "
        "validators allocate their stake to different miners over time. "
        "Each line represents a validator's weight on a specific miner."
    )


def _prepare_validator_server_weights_subplots_data(
    validators: list[str],
    weights_epochs: list[torch.Tensor],
    servers: list[str],
    num_epochs: int,
    epochs_padding: int = 0,
) -> dict | None:
    """
    Prepare common data for validator server weights subplots plotting.

    Returns:
        dict with plotting data or None if nothing to plot
    """
    from .simulation_utils import _slice_tensors

    plot_epochs = num_epochs - epochs_padding
    if plot_epochs <= 0:
        return None

    weights_epochs = _slice_tensors(
        *weights_epochs,
        num_validators=len(validators),
        num_servers=len(servers)
    )

    # Build data matrix: [server][validator][epoch_values]
    data_matrix = []
    for idx_s, server_name in enumerate(servers):
        server_data = []
        for idx_v, validator in enumerate(validators):
            y_values = [
                float(weights_epochs[epoch][idx_v][idx_s].item())
                for epoch in range(epochs_padding, num_epochs)
            ]
            server_data.append(y_values)
        data_matrix.append(server_data)

    return {
        'validators': validators,
        'servers': servers,
        'num_epochs': num_epochs,
        'plot_epochs': plot_epochs,
        'data_matrix': data_matrix,
        'x': list(range(plot_epochs)),
        'y_range': [0, 1.05]
    }


def _calculate_total_dividends(
    validators: list[str],
    dividends_per_validator: dict[str, list[float]],
    base_validator: str,
    num_epochs: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Calculates total dividends and percentage differences from a base validator."""

    total_dividends: dict[str, float] = {}
    for validator in validators:
        divs: list[float] = dividends_per_validator.get(validator, [])
        total_dividend = sum(divs[:num_epochs])
        total_dividends[validator] = total_dividend

    base_dividend = total_dividends.get(base_validator, None)
    if base_dividend is None or base_dividend == 0.0:
        logger.warning(
            f"Warning: Base validator '{base_validator}' has zero or missing total dividends."
        )
        base_dividend = 1e-6

    percentage_diff_vs_base: dict[str, float] = {}
    for validator, total_dividend in total_dividends.items():
        if validator == base_validator:
            percentage_diff_vs_base[validator] = 0.0
        else:
            percentage_diff = ((total_dividend - base_dividend) / base_dividend) * 100.0
            percentage_diff_vs_base[validator] = percentage_diff

    return total_dividends, percentage_diff_vs_base


def _prepare_dividends_data(
    num_epochs: int,
    validators: list[str],
    dividends_per_validator: dict[str, list[float]],
    case: BaseCase,
) -> dict | None:
    """
    Prepare common data for dividends plotting.

    Returns:
        dict with plotting data or None if nothing to plot
    """

    top_vals = getattr(case, "requested_validators", [])
    if top_vals:
        plot_validator_names = top_vals.copy()
    else:
        plot_validator_names = validators.copy()

    if case.base_validator not in plot_validator_names:
        plot_validator_names.append(case.base_validator)

    # Calculate total dividends and percentage differences
    total_dividends, percentage_diff_vs_base = _calculate_total_dividends(
        validators,
        dividends_per_validator,
        case.base_validator,
        num_epochs,
    )

    num_epochs_calculated = None
    x = None
    series_data = []

    for idx, validator in enumerate(plot_validator_names):
        if validator not in dividends_per_validator:
            continue

        dividends = dividends_per_validator[validator]
        dividends_array = np.array(dividends, dtype=float)

        if num_epochs_calculated is None:
            num_epochs_calculated = len(dividends_array)
            x = np.arange(num_epochs_calculated)

        # Prepare shifted x values for better visibility
        delta = 0.05
        x_shifted = x + idx * delta

        # Format labels
        total_dividend = total_dividends.get(validator, 0.0)
        percentage_diff = percentage_diff_vs_base.get(validator, 0.0)

        if abs(total_dividend) < 1e-6:
            total_dividend_str = f"{total_dividend:.3e}"
        else:
            total_dividend_str = f"{total_dividend:.6f}"

        if abs(percentage_diff) < 1e-12:
            percentage_str = "(Base)"
        elif percentage_diff > 0:
            percentage_str = f"(+{percentage_diff:.1f}%)"
        else:
            percentage_str = f"({percentage_diff:.1f}%)"

        # Get validator display name
        if hasattr(case, 'hotkey_label_map'):
            display_name = case.hotkey_label_map.get(validator, validator)
        else:
            display_name = validator
        label = f"{display_name}: Total={total_dividend_str} {percentage_str}"

        series_data.append({
            'validator': validator,
            'data': dividends_array[:num_epochs],
            'x_shifted': x_shifted.tolist(),
            'x': x.tolist(),  # Also provide non-shifted version
            'label': label,
            'idx': idx,
            'total_dividend': total_dividend,
            'percentage_diff': percentage_diff,
        })

    if not series_data:
        return None

    return {
        'series_data': series_data,
        'x': x.tolist() if x is not None else [],
        'num_epochs_calculated': num_epochs_calculated,
        'plot_validator_names': plot_validator_names,
        'total_dividends': total_dividends,
        'percentage_diff_vs_base': percentage_diff_vs_base,
    }


def _prepare_bonds_data(
    bonds_per_epoch: list[torch.Tensor],
    validators: list[str],
    servers: list[str],
    normalize: bool,
) -> list[list[list[float]]]:
    """Prepares bond data for plotting, normalizing if specified."""

    num_epochs = len(bonds_per_epoch)
    bonds_data: list[list[list[float]]] = []
    for idx_s, _ in enumerate(servers):
        server_bonds: list[list[float]] = []
        for idx_v, _ in enumerate(validators):
            validator_bonds = [
                float(bonds_per_epoch[epoch][idx_v, idx_s].item())
                for epoch in range(num_epochs)
            ]
            server_bonds.append(validator_bonds)
        bonds_data.append(server_bonds)

    if normalize:
        for idx_s in range(len(servers)):
            for epoch in range(num_epochs):
                epoch_bonds = bonds_data[idx_s]
                values = [epoch_bonds[idx_v][epoch] for idx_v in range(len(validators))]
                total = sum(values)
                if total > 1e-12:
                    for idx_v in range(len(validators)):
                        epoch_bonds[idx_v][epoch] /= total

    return bonds_data


def _prepare_validator_server_weights_data(
    validators: list[str],
    weights_epochs: list[torch.Tensor],
    servers: list[str],
    num_epochs: int,
) -> dict | None:
    """Prepare data for validator server weights plotting with custom y-axis logic."""
    from .simulation_utils import _slice_tensors

    if num_epochs <= 0:
        return None

    weights_epochs = _slice_tensors(*weights_epochs, num_validators=len(validators), num_servers=len(servers))

    # Extract all y values for tick calculation
    y_values_all: list[float] = [
        float(weights_epochs[epoch][idx_v][1].item())
        for idx_v in range(len(validators))
        for epoch in range(num_epochs)
    ]
    unique_y_values = sorted(set(y_values_all))

    # Calculate custom y-axis ticks
    y_tick_positions, y_tick_labels = _calculate_custom_y_ticks(
        unique_y_values, servers
    )

    # Prepare series data
    series_data = []
    for idx_v, validator in enumerate(validators):
        y_values = [
            float(weights_epochs[epoch][idx_v][1].item())
            for epoch in range(num_epochs)
        ]

        series_data.append({
            'validator': validator,
            'data': y_values,
            'idx': idx_v,
            'label': validator,
        })

    # Determine adaptive height based on tick count
    fig_height_scale = 1 if len(y_tick_positions) <= 2 else 3

    return {
        'series_data': series_data,
        'x': list(range(num_epochs)),
        'y_tick_positions': y_tick_positions,
        'y_tick_labels': y_tick_labels,
        'fig_height_scale': fig_height_scale,
        'y_range': [-0.05, 1.05],
    }


def _calculate_custom_y_ticks(unique_y_values: list[float], servers: list[str]) -> tuple[list[float], list[str]]:
    """Calculate custom y-axis tick positions and labels for server weights."""
    min_label_distance = 0.05
    close_to_server_threshold = 0.02

    def is_round_number(y: float) -> bool:
        return abs((y * 20) - round(y * 20)) < 1e-6

    # Start with server endpoints
    y_tick_positions: list[float] = [0.0, 1.0]
    y_tick_labels: list[str] = [servers[0], servers[1]]

    for y in unique_y_values:
        if y in [0.0, 1.0]:
            continue
        if (
            abs(y - 0.0) < close_to_server_threshold
            or abs(y - 1.0) < close_to_server_threshold
        ):
            continue

        # Check if we should add this tick
        should_add = all(
            abs(y - existing_y) >= min_label_distance
            for existing_y in y_tick_positions
        )

        if should_add:
            y_tick_positions.append(y)
            y_percentage = y * 100
            label = (
                f"{y_percentage:.0f}%"
                if float(y_percentage).is_integer()
                else f"{y_percentage:.1f}%"
            )
            y_tick_labels.append(label)

    # Sort ticks by position
    ticks = list(zip(y_tick_positions, y_tick_labels))
    ticks.sort(key=lambda x: x[0])
    y_tick_positions, y_tick_labels = map(list, zip(*ticks))

    return y_tick_positions, y_tick_labels


def _prepare_incentives_data(
    servers: list[str],
    server_incentives_per_epoch: list[torch.Tensor],
    num_epochs: int,
    case: BaseCase = None,
) -> dict | None:
    """Prepare common data for server incentives plotting."""

    if num_epochs <= 0 or not servers or not server_incentives_per_epoch:
        return None

    x = list(range(num_epochs))
    series_data = []

    for idx_s, server in enumerate(servers):
        incentives = [
            float(server_incentive[idx_s].item())
            for server_incentive in server_incentives_per_epoch
        ]

        # Create label (use hotkey mapping if available)
        if case and hasattr(case, 'hotkey_label_map'):
            label = case.hotkey_label_map.get(server, server[:12] + '...' if len(server) > 12 else server)
        else:
            label = server[:12] + '...' if len(server) > 12 else server

        series_data.append({
            'server': server,
            'data': incentives,
            'x': x,
            'label': label,
            'idx': idx_s
        })

    return {
        'series_data': series_data,
        'x': x,
        'num_epochs': num_epochs
    }


def _get_incentives_description() -> str:
    """Get the description for server incentives visualization."""
    return (
        "'Server Incentives' shows how much incentive (reward) each server "
        "receives over time. Incentives typically range from 0 to 1 and represent "
        "the server's share of the total incentive pool at each epoch."
    )
