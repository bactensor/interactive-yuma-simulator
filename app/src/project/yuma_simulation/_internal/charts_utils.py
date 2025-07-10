"""
This module provides utilities for calculating and visualizing simulation results.
It includes functions for generating plots, calculating dividends, and preparing data for bonds and incentives.
"""

import base64
import io
import logging
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.set_loglevel("warning")

import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import textwrap
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
from project.yuma_simulation._internal.cases import BaseCase, MetagraphCase
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FuncFormatter

from project.yuma_simulation._internal.charts_data import _calculate_total_dividends
from project.yuma_simulation._internal.charts_plotly import (
    plot_relative_dividends_plotly,
    plot_bonds_metagraph_dynamic_plotly,
    plot_validator_server_weights_subplots_dynamic_plotly,
    plot_validator_server_weights_subplots_plotly,
    plot_dividends_plotly,
    plot_bonds_plotly,
    plot_validator_server_weights_plotly,
    plot_incentives_plotly,
)
from project.yuma_simulation._internal.charts_matplotlib import (
    _plot_relative_dividends_matplotlib,
    _plot_bonds_metagraph_dynamic_matplotlib,
    _plot_validator_server_weights_subplots_dynamic_matplotlib,
    _plot_validator_server_weights_subplots_matplotlib,
    _plot_dividends_matplotlib,
    _plot_bonds_matplotlib,
    _plot_validator_server_weights_matplotlib,
    _plot_incentives_matplotlib,
    _plot_relative_dividends_comparisson,
)

logger = logging.getLogger(__name__)


def _calculate_total_dividends_with_frames(
    validator_dividends: list[float],
    num_epochs: int,
    epochs_window: int,
    use_relative: bool = False
) -> tuple[list[float], float]:
    """
    Returns a tuple of:
      1) A list of "frame" values over consecutive windows of length `epochs_window`.
      2) The overall "total" (sum for absolute, or average for relative).

    If `use_relative=False` (default), each frame is summed:
      e.g. [sum of epochs 0..9, sum of epochs 10..19, ...]

    If `use_relative=True`, each frame is an average:
      e.g. [avg of epochs 0..9, avg of epochs 10..19, ...]
      And the overall total is the avg of all truncated_divs.

    For example:
      If num_epochs=40, epochs_window=10 => 4 frames (each covering 10 epochs).
      With `use_relative=False`, we sum each window.
      With `use_relative=True`, we average each window.
    """

    # Truncate any extra dividends if validator_dividends is longer than num_epochs
    truncated_divs = validator_dividends[:num_epochs]

    frames_values = []
    for start_idx in range(0, num_epochs, epochs_window):
        end_idx = min(start_idx + epochs_window, num_epochs)
        chunk = truncated_divs[start_idx:end_idx]

        if use_relative:
            val = sum(chunk) / len(chunk)
        else:
            val = sum(chunk)

        frames_values.append(val)

    if use_relative:
        total_value = sum(truncated_divs) / len(truncated_divs)
    else:
        total_value = sum(truncated_divs)

    return frames_values, total_value


def _plot_dividends(
    num_epochs: int,
    validators: list[str],
    dividends_per_validator: dict[str, list[float]],
    case_name: str,
    case: BaseCase,
    to_base64: bool = False,
    engine: str = "matplotlib",  # or "plotly"
    **kwargs,
) -> str | None:
    """
    Plot dividends over epochs using specified engine.

    Args:
        engine: "matplotlib" or "plotly"
        to_base64: Only used with matplotlib
        **kwargs: Additional arguments passed to specific plotting function
    """
    if engine == "matplotlib":
        return _plot_dividends_matplotlib(
            num_epochs, validators, dividends_per_validator,
            case_name, case, to_base64, **kwargs
        )
    elif engine == "plotly":
        return plot_dividends_plotly(
            num_epochs, validators, dividends_per_validator,
            case_name, case, **kwargs
        )
    else:
        raise ValueError(f"Unknown plotting engine: {engine}")



def _plot_relative_dividends(
    validators_relative_dividends: dict[str, list[float]],
    case_name: str,
    case: BaseCase,
    num_epochs: int,
    epochs_padding: int = 0,
    engine: str = "matplotlib",  # or "plotly"
    **kwargs,
) -> str | None:
    """
    Plot relative dividends using specified engine.

    Args:
        engine: "matplotlib" or "plotly"
        **kwargs: Additional arguments passed to specific plotting function
    """
    if engine == "matplotlib":
        return _plot_relative_dividends_matplotlib(
            validators_relative_dividends, case_name, case, num_epochs, epochs_padding, **kwargs
        )
    elif engine == "plotly":
        return plot_relative_dividends_plotly(
            validators_relative_dividends, case_name, case, num_epochs, epochs_padding, **kwargs
        )
    else:
        raise ValueError(f"Unknown plotting engine: {engine}")



def _plot_bonds(
    num_epochs: int,
    validators: list[str],
    servers: list[str],
    bonds_per_epoch: list[torch.Tensor],
    case_name: str,
    to_base64: bool = False,
    normalize: bool = False,
    engine: str = "matplotlib",
) -> str | None:
    """Generates a plot of bonds per server for each validator."""

    if engine == "plotly":
        return plot_bonds_plotly(
            num_epochs, validators, servers, bonds_per_epoch,
            case_name, normalize
        )
    else:
        return _plot_bonds_matplotlib(
            num_epochs, validators, servers, bonds_per_epoch,
            case_name, to_base64, normalize
        )


def _plot_bonds_metagraph_dynamic(
    case: MetagraphCase,
    bonds_per_epoch: list[torch.Tensor],
    default_miners: list[str],
    case_name: str,
    engine: str = "matplotlib",  # or "plotly"
    **kwargs,
) -> str | None:
    """
    Plot bonds metagraph using specified engine.

    Args:
        engine: "matplotlib" or "plotly"
        **kwargs: Additional arguments passed to specific plotting function
    """
    if engine == "matplotlib":
        return _plot_bonds_metagraph_dynamic_matplotlib(
            case, bonds_per_epoch, default_miners, case_name, **kwargs
        )
    elif engine == "plotly":
        return plot_bonds_metagraph_dynamic_plotly(
            case, bonds_per_epoch, default_miners, case_name, **kwargs
        )
    else:
        raise ValueError(f"Unknown plotting engine: {engine}")


def _plot_validator_server_weights(
    validators: list[str],
    weights_epochs: list[torch.Tensor],
    servers: list[str],
    num_epochs: int,
    case_name: str,
    engine: str = "matplotlib",
    to_base64: bool = False,
    **kwargs,
) -> str | None:
    """
    Unified interface for plotting validator weights across servers over epochs.

    Args:
        validators: List of validator identifiers
        weights_epochs: List of weight tensors per epoch
        servers: List of server identifiers
        num_epochs: Number of epochs to plot
        case_name: Name of the case for the title
        engine: Rendering engine ("matplotlib" or "plotly")
        to_base64: Whether to return base64 encoded image (matplotlib only)
        **kwargs: Additional arguments passed to the rendering engine

    Returns:
        For matplotlib: base64 string if to_base64=True, otherwise None
        For plotly: HTML string with embedded chart
    """
    if engine.lower() == "plotly":
        return plot_validator_server_weights_plotly(
            validators, weights_epochs, servers, num_epochs, case_name, **kwargs
        )
    else:
        return _plot_validator_server_weights_matplotlib(
            validators, weights_epochs, servers, num_epochs, case_name, to_base64, **kwargs
        )


def _plot_validator_server_weights_subplots(
    validators: list[str],
    weights_epochs: list[torch.Tensor],
    servers: list[str],
    num_epochs: int,
    case_name: str,
    to_base64: bool = False,
    epochs_padding: int = 0,
    engine: str = "matplotlib",  # or "plotly"
    **kwargs,
) -> str | None:
    """
    Plot validator server weights in subplots using specified engine.

    Args:
        validators: List of validator identifiers
        weights_epochs: List of weight tensors per epoch
        servers: List of server identifiers
        num_epochs: Number of epochs to plot
        case_name: Name for the case/simulation
        to_base64: Whether to return matplotlib plot as base64 (matplotlib only)
        epochs_padding: Number of epochs to skip from beginning
        engine: "matplotlib" or "plotly"
        **kwargs: Additional arguments passed to specific plotting function
    """
    if engine == "matplotlib":
        return _plot_validator_server_weights_subplots_matplotlib(
            validators, weights_epochs, servers, num_epochs, case_name,
            to_base64, epochs_padding, **kwargs
        )
    elif engine == "plotly":
        return plot_validator_server_weights_subplots_plotly(
            validators, weights_epochs, servers, num_epochs, case_name,
            epochs_padding, **kwargs
        )
    else:
        raise ValueError(f"Unknown plotting engine: {engine}")


def _plot_validator_server_weights_subplots_dynamic(
    case: MetagraphCase,
    default_miners: list[str],
    case_name: str = "",
    epochs_padding: int = 0,
    engine: str = "matplotlib",  # or "plotly"
    **kwargs,
) -> str | None:
    """
    Plot validator server weights using specified engine.

    Args:
        engine: "matplotlib" or "plotly"
        **kwargs: Additional arguments passed to specific plotting function
    """
    if engine == "matplotlib":
        return _plot_validator_server_weights_subplots_dynamic_matplotlib(
            case, default_miners, epochs_padding, **kwargs
        )
    elif engine == "plotly":
        return plot_validator_server_weights_subplots_dynamic_plotly(
            case, default_miners, case_name, epochs_padding, **kwargs
        )
    else:
        raise ValueError(f"Unknown plotting engine: {engine}")


def _plot_incentives(
    servers: list[str],
    server_incentives_per_epoch: list[torch.Tensor],
    num_epochs: int,
    case_name: str,
    case: BaseCase = None,
    to_base64: bool = False,
    engine: str = "matplotlib",
    **kwargs,
) -> str | None:
    """
    Router function for plotting server incentives over epochs.

    Args:
        servers: List of server identifiers
        server_incentives_per_epoch: List of tensors containing incentive values per epoch
        num_epochs: Total number of epochs to plot
        case_name: Name/description of the case being plotted
        case: Optional BaseCase object for enhanced labeling
        to_base64: Whether to return matplotlib plot as base64 string
        engine: Rendering engine ("matplotlib" or "plotly")
        **kwargs: Additional arguments passed to the plotting functions

    Returns:
        For matplotlib: base64 string if to_base64=True, None otherwise
        For plotly: HTML string with embedded chart
    """

    if engine.lower() == "plotly":
        return plot_incentives_plotly(
            servers=servers,
            server_incentives_per_epoch=server_incentives_per_epoch,
            num_epochs=num_epochs,
            case_name=case_name,
            case=case,
            **kwargs
        )
    elif engine.lower() == "matplotlib":
        return _plot_incentives_matplotlib(
            servers=servers,
            server_incentives_per_epoch=server_incentives_per_epoch,
            num_epochs=num_epochs,
            case_name=case_name,
            case=case,
            to_base64=to_base64,
        )
    else:
        raise ValueError(f"Unsupported engine: {engine}. Use 'matplotlib' or 'plotly'.")


def wrap_raise(fun):
    import traceback
    def __inner(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except Exception as exc:
            traceback.print_exception(exc)
            raise
    return __inner


@wrap_raise
def _generate_chart_for_type(
    chart_type: str,
    case: BaseCase,
    final_case_name: str,
    simulation_results: tuple | None = None,
    to_base64: bool = True,
    epochs_padding: int = 0,
    engine: str = 'matplotlib',
) -> str:
    """
    Dispatches to the correct plotting function based on the chart type.
    For types that need simulation results, the tuple is unpacked as needed.
    """
    if chart_type == "weights":
        return _plot_validator_server_weights(
            validators=case.validators,
            weights_epochs=case.weights_epochs,
            servers=case.servers,
            num_epochs=case.num_epochs,
            case_name=final_case_name,
            to_base64=to_base64,
            engine=engine,
        )
    elif chart_type == "weights_subplots":
        return  _plot_validator_server_weights_subplots(
            validators=case.validators,
            weights_epochs=case.weights_epochs,
            servers=case.servers,
            num_epochs=case.num_epochs,
            case_name=final_case_name,
            to_base64=to_base64,
            engine=engine,
        )
    elif chart_type == "dividends":
        dividends_per_validator, *_ = simulation_results
        return _plot_dividends(
            num_epochs=case.num_epochs,
            validators=case.validators,
            dividends_per_validator=dividends_per_validator,
            case_name=final_case_name,
            case=case,
            to_base64=to_base64,
            engine=engine,
        )
    elif chart_type == "relative_dividends":
        _, validators_relative_dividends, *_ = simulation_results
        return _plot_relative_dividends(
            validators_relative_dividends=validators_relative_dividends,
            case_name=final_case_name,
            case=case,
            num_epochs=case.num_epochs,
            epochs_padding=epochs_padding,
            to_base64=to_base64,
            engine=engine,
        )
    elif chart_type == "bonds":
        _, _, bonds_per_epoch, *_ = simulation_results
        return _plot_bonds(
            num_epochs=case.num_epochs,
            validators=case.validators,
            servers=case.servers,
            bonds_per_epoch=bonds_per_epoch,
            case_name=final_case_name,
            to_base64=to_base64,
            engine=engine,
        )
    elif chart_type == "normalized_bonds":
        _, _, bonds_per_epoch, *_ = simulation_results
        return _plot_bonds(
            num_epochs=case.num_epochs,
            validators=case.validators,
            servers=case.servers,
            bonds_per_epoch=bonds_per_epoch,
            case_name=final_case_name,
            to_base64=to_base64,
            normalize=True,
            engine=engine,
        )
    elif chart_type == "incentives":
        *_, server_incentives_per_epoch = simulation_results
        return _plot_incentives(
            servers=case.servers,
            server_incentives_per_epoch=server_incentives_per_epoch,
            num_epochs=case.num_epochs,
            case_name=final_case_name,
            to_base64=to_base64,
            engine=engine,
        )
    else:
        raise ValueError("Invalid chart type.")


def _construct_relative_dividends_table(
    relative_dividends_by_version: dict[str, dict[str, list[float]]],
    validators: list[str],
    diff_versions: tuple[str, str] | None = None,
    epochs_padding: int = 0,
    num_epochs: int = 0,
    alpha_tao_ratio: float = 1.0,
) -> pd.DataFrame:
    """
        Constructs a DataFrame comparing scaled relative dividends across Yuma versions:
      - '<version>_mean': mean of (padded) series multiplied by 361 * 0.41 * num_epochs * alpha_tao_ratio
      - if diff_versions is provided, 'diff_<vA>_<vB>': scaled_mean_vA - scaled_mean_vB

    """
    effective_epochs = num_epochs - epochs_padding
    if effective_epochs < 0:
        effective_epochs = 0

    factor = 361 * 0.41 * effective_epochs * alpha_tao_ratio

    rows: list[dict[str, str]] = []
    for v in validators:
        row: dict[str, str] = {"validator": v}
        scaled_means: dict[str, float] = {}
        raw_means: dict[str, float]    = {}

        for version, divs in relative_dividends_by_version.items():
            series = divs.get(v, [])
            trimmed = series[epochs_padding:] if len(series) > epochs_padding else []

            arr = np.array([x if (x is not None) else np.nan for x in trimmed], dtype=float)

            if arr.size > 0:
                base_mean = float(np.nanmean(arr))
            else:
                base_mean = 0.0

            raw_means[version] = base_mean

            scaled = base_mean * factor
            scaled_means[version] = scaled

            scaled_str  = f"{scaled:+.2f} τ"
            raw_pct_str = f"{(base_mean * 100):+.2f}%"
            cell_text   = f"{scaled_str} ({raw_pct_str})"
            row[version] = cell_text

        if diff_versions is not None:
            vA, vB = diff_versions
            diff_col = f"diff_{vA}_{vB}"

            raw_diff    = raw_means.get(vA, 0.0) - raw_means.get(vB, 0.0)
            scaled_diff = scaled_means.get(vA, 0.0) - scaled_means.get(vB, 0.0)

            scaled_str_diff = f"{scaled_diff:+.2f} τ"
            raw_pct_diff    = f"{(raw_diff * 100):+.2f}%"
            cell_diff_text  = f"{scaled_str_diff} ({raw_pct_diff})"
            row[diff_col]   = cell_diff_text

        rows.append(row)

    df = pd.DataFrame(rows).set_index("validator")
    return df

def _generate_relative_dividends_summary_html(
    relative_dividends_by_version: dict[str, dict[str, list[float]]],
    top_validators: list[str],
    diff_versions: tuple[str, str] | None = None,
    epochs_padding: int = 0,
    num_epochs: int = 0,
    alpha_tao_ratio: float = 1.0,
    label_map: dict[str, str] | None = None,
) -> str:
    """
    Build a Bootstrap‐styled HTML table for the scaled relative dividends
    of `top_validators` across Yuma versions, with optional display name mapping.

    Scaled means by 361 * 0.41 * (num_epochs - epochs_padding) * alpha_tao_ratio.
    If `diff_versions` is provided, includes a diff column 'diff_vA_vB'.
    If `label_map` is given, uses that to replace validator IDs in the index.
    """

    df = _construct_relative_dividends_table(
        relative_dividends_by_version,
        top_validators,
        diff_versions=diff_versions,
        epochs_padding=epochs_padding,
        num_epochs=num_epochs,
        alpha_tao_ratio=alpha_tao_ratio,
    )

    if label_map is not None:
        df.index = [
            label_map[v] + f'<span style="font-size: 0.66rem;"><br>{v}</span>' if v in label_map else v
            for v in df.index
        ]

    tao_icon = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'viewBox="0 0 467.715 408.195" '
        'preserveAspectRatio="xMaxYMax slice" '
        'fill="black" '
        'width="1.2em" height="1.2em" '
        'style="display:inline-block; '
            'vertical-align:middle; '
            'transform:translateY(-0.1em); '
            'margin-left:0.05em;">'
        '<path d="M271.215,286.17c-11.76,7.89-23.85,8.31-36.075,2.865c-11.43-5.1-16.695-14.64-16.725-26.955c-0.09-35.49-0.03-70.98-0.03-106.485'
        'c0-2.16,0-4.305,0-7.08c-22.05,0-43.815,0-65.52,0c-1.38-13.335,9.93-27.885,22.83-29.73c5.4-0.765,10.89-1.185,16.35-1.2'
        'c38.85-0.105,77.685-0.06,116.535-0.06c2.13,0,4.275,0,6.42,0c-0.18,16.5-11.715,30.15-30.33,30.63c-18.915,0.495-37.845,0.105-56.985,0.435'
        'c9.9,4.125,17.7,10.455,21.255,20.7c1.5,4.335,2.4,9.105,2.445,13.68c0.225,26.415-0.15,52.845,0.195,79.26C251.805,279.99,258.36,282.9,271.215,286.17z"/>'
        '</svg>'
    )

    for col in df.columns:
        df[col] = df[col].map(lambda x: (
            '<span style="white-space:nowrap; line-height:1em;">'
            + (x if isinstance(x, str) else f"{float(x):+.2f} τ")
            + tao_icon
            + '</span>'
        ))

    table_html = df.to_html(
        classes="table table-striped table-bordered",
        border=0,
        index=True,
        escape=False,
    )

    title_html = '<h4 class="mt-4">Relative Dividends Summary</h4>'
    desc_html = (
        f'<p class="mb-3 plotly-chart-html-description">'
        f"This table shows, for each of your top validators, the mean “relative dividend” "
        f"across different Yuma versions, after scaling by "
        f"<code>361 × 0.41 × (epochs-number) × alpha-tao-ratio</code>. "
        f"(Higher numbers → better performance.)"
        f'</p>'
    )

    return f"{title_html}{desc_html}{table_html}"


def _pick_default_miners(
    hotkeys_incentive_over_epochs: dict[str, list[float]]
) -> list[str]:
    """
    Selects four default miners based on their incentive time-series:
      1. Top 2 by total incentive received across all epochs.
      2. Next 2 by incentive spread (max − min).
         If the top‐2 spread hotkeys exactly match the top‐2 total hotkeys,
         instead grab the 3rd and 4th by spread.
    Returns a list of four hotkey IDs.
    """
    total_received = {
        hk: sum(series)
        for hk, series in hotkeys_incentive_over_epochs.items()
    }
    spread = {
        hk: (max(series) - min(series)) if series else 0.0
        for hk, series in hotkeys_incentive_over_epochs.items()
    }

    sorted_by_total = [
        hk for hk, _ in sorted(
            total_received.items(),
            key=lambda kv: kv[1],
            reverse=True
        )
    ]
    sorted_by_spread = [
        hk for hk, _ in sorted(
            spread.items(),
            key=lambda kv: kv[1],
            reverse=True
        )
    ]

    total_count = 4
    spread_count = 4

    top_by_total = sorted_by_total[:total_count]
    top_total_set = set(top_by_total)
    top_by_spread = []
    for hk in sorted_by_spread:
        if hk not in top_total_set:
            top_by_spread.append(hk)
        if len(top_by_spread) >= spread_count:
            break

    return top_by_total + top_by_spread
