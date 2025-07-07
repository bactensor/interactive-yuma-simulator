import matplotlib
matplotlib.use("Agg")
matplotlib.set_loglevel("warning")

import matplotlib.pyplot as plt
import textwrap
import numpy as np
import io
import base64

from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
from project.yuma_simulation._internal.cases import BaseCase, MetagraphCase
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FuncFormatter

import torch
import math

from project.yuma_simulation._internal.charts_data import (
    _prepare_relative_dividends_data,
    _get_relative_dividends_description_and_formula,
    _prepare_bonds_metagraph_data,
    _get_bonds_description_and_labels,
    _prepare_validator_server_weights_subplots_dynamic_data,
    _get_validator_weights_description,
    _prepare_validator_server_weights_subplots_data,
    _prepare_dividends_data,
    _prepare_bonds_data,
    _prepare_validator_server_weights_data,
    _prepare_incentives_data,
)

# TODO: refactor to not use it?
from project.yuma_simulation._internal.charts_data import (
    _get_validator_styles,
    _compute_mean,
)


def _set_default_xticks(ax: Axes, num_epochs: int) -> None:
    tick_locs = [0, 1, 2] + list(range(5, num_epochs, 5))
    tick_labels = [str(i) for i in tick_locs]
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels, fontsize=8)


def _plot_to_base64(fig: plt.Figure) -> str:
    """Converts a Matplotlib figure to a Base64-encoded string."""

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", dpi=100)
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode("ascii")
    buf.close()
    plt.close(fig)
    return f'<img src="data:image/png;base64,{encoded_image}" height:auto;" draggable="false">'


def _plot_relative_dividends_matplotlib(
    validators_relative_dividends: dict[str, list[float]],
    case_name: str,
    case: BaseCase,
    num_epochs: int,
    epochs_padding: int = 0,
    to_base64: bool = False
) -> str | None:
    plt.close("all")

    # Use common data preparation
    data = _prepare_relative_dividends_data(
        validators_relative_dividends, case, num_epochs, epochs_padding
    )
    if data is None:
        logger.warning("Nothing to plot (padding/empty data).")
        return None

    # Get description and formula
    description, formula_latex, _ = _get_relative_dividends_description_and_formula()

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[1, 4], figure=fig)

    # Text subplot
    ax_text = fig.add_subplot(gs[0])
    ax_text.axis("off")
    wrapped_para = textwrap.fill(description, width=160)
    full_text = wrapped_para + "\n\n" + r"$\text{Relative Dividend} =$ " + formula_latex
    ax_text.text(0.5, 0.5, full_text, ha="center", va="center", fontsize=12, wrap=False)

    # Main plot
    ax = fig.add_subplot(gs[1])

    for series_info in data['series_data']:
        x_shifted = np.array(series_info['x_shifted'])
        ax.plot(
            x_shifted, series_info['data'],
            label=series_info['label'],
            alpha=0.7,
            linestyle=series_info['line_style'],
            marker=series_info['marker'],
            markersize=series_info['marker_size'],
            markeredgewidth=series_info['marker_edge_width']
        )

    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.7)
    _set_default_xticks(ax, data['plot_epochs'])
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Relative Dividend (%)", fontsize=12)

    legend = ax.legend()
    for text in legend.get_texts():
        if text.get_text().startswith(case.shift_validator_hotkey):
            text.set_fontweight("bold")

    ax.grid(True)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.1f}%"))
    fig.suptitle(f"Validators Relative Dividends\n{case_name}", fontsize=16)

    if to_base64:
        return _plot_to_base64(fig)
    else:
        plt.show()
        plt.close(fig)
        return None


def _plot_bonds_metagraph_dynamic_matplotlib(
    case: MetagraphCase,
    bonds_per_epoch: list[torch.Tensor],
    default_miners: list[str],
    case_name: str,
    to_base64: bool = False,
    normalize: bool = False,
    legend_validators: list[str] | None = None,
    epochs_padding: int = 0,
) -> str | None:

    # Use common data preparation
    data = _prepare_bonds_metagraph_data(
        case, bonds_per_epoch, default_miners, normalize, epochs_padding
    )
    if data is None:
        logger.warning("Nothing to plot (padding >= total_epochs).")
        return None

    # Get description and labels
    description, ylabel, title_suffix = _get_bonds_description_and_labels(normalize)

    # Layout constants
    CHART_WIDTH = 7.0
    CHART_HEIGHT = 5.0
    TEXT_BLOCK_H = 2.0
    COLS = 2

    num_charts = len(data['subset_m'])
    rows = math.ceil(num_charts / COLS)

    fig_w = CHART_WIDTH * COLS
    fig_h = TEXT_BLOCK_H + (CHART_HEIGHT * rows)

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)

    chart_block_h = CHART_HEIGHT * rows

    outer_gs = GridSpec(
        nrows=2,
        ncols=1,
        height_ratios=[TEXT_BLOCK_H, chart_block_h],
        hspace=0.1,
        figure=fig
    )

    top_gs = GridSpecFromSubplotSpec(
        nrows=3,
        ncols=1,
        subplot_spec=outer_gs[0],
        height_ratios=[0.5, 1.0, 0.5],
        hspace=0.0
    )

    # Title
    ax_title = fig.add_subplot(top_gs[0])
    ax_title.axis("off")
    title_str = f"Validators bonds per Miner{title_suffix}\n{case_name}"
    ax_title.text(0.5, 0.5, title_str, ha="center", va="center", fontsize=14)

    # Description
    ax_para = fig.add_subplot(top_gs[1])
    ax_para.axis("off")
    wrapped = textwrap.fill(description, width=140)
    ax_para.text(0.5, 0.5, wrapped, ha="center", va="center", fontsize=11, wrap=False)

    # Legend placeholder
    ax_legend = fig.add_subplot(top_gs[2])
    ax_legend.axis("off")

    # Chart grid
    inner_gs = GridSpecFromSubplotSpec(
        nrows=rows,
        ncols=COLS,
        subplot_spec=outer_gs[1],
        wspace=0.3,
        hspace=0.4
    )

    styles = _get_validator_styles(data['validator_keys'])
    handles = []
    labels = []

    # Setup ticks
    ticks = list(range(0, data['plot_epochs'], 5))
    if (data['plot_epochs'] - 1) not in ticks:
        ticks.append(data['plot_epochs'] - 1)
    tick_labels = [str(t) for t in ticks]

    # Plot each miner's chart
    for i_miner, miner in enumerate(data['subset_m']):
        r, c = divmod(i_miner, COLS)
        ax = fig.add_subplot(inner_gs[r, c])

        for j, val in enumerate(data['subset_v']):
            ls, mk, ms, mew = styles[val]
            line, = ax.plot(
                data['x'], data['plot_data'][i_miner][j],
                linestyle=ls,
                marker=mk,
                markersize=ms,
                markeredgewidth=mew,
                linewidth=2,
                alpha=0.7
            )
            if i_miner == 0 and (legend_validators or data['subset_v']) and (val in (legend_validators or data['subset_v'])):
                handles.append(line)
                labels.append(case.hotkey_label_map.get(val, val))

        ax.set_title(miner, fontsize=10)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Epoch")
        if r == 0 and c == 0:
            ax.set_ylabel(ylabel)
        ax.grid(True)
        if normalize:
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(bottom=0)

    # Hide empty subplots
    total_slots = rows * COLS
    for idx in range(num_charts, total_slots):
        r, c = divmod(idx, COLS)
        ax_empty = fig.add_subplot(inner_gs[r, c])
        ax_empty.set_visible(False)

    # Add legend
    ncol = min(len(labels), 4)
    ax_legend.legend(
        handles,
        labels,
        loc="center",
        ncol=ncol,
        frameon=False,
        fontsize="small",
        handletextpad=0.3,
        columnspacing=0.5
    )

    fig.tight_layout(w_pad=0.3, h_pad=0.4)

    if to_base64:
        return _plot_to_base64(fig)

    plt.show()
    plt.close(fig)
    return None


def _plot_validator_server_weights_subplots_dynamic_matplotlib(
    case: MetagraphCase,
    default_miners: list[str],
    epochs_padding: int = 0,
    to_base64: bool = False,
) -> str | None:
    """
    Dynamic version for metagraph-based weights using matplotlib
    """
    # Use common data preparation
    data = _prepare_validator_server_weights_subplots_dynamic_data(case, default_miners, epochs_padding)
    if data is None:
        print("Nothing to plot (padding >= total_epochs).")
        return None

    # Get description
    description = _get_validator_weights_description()

    # Layout constants
    CHART_WIDTH = 7
    CHART_HEIGHT = 5
    TEXT_BLOCK_H = 2
    COLS = 2

    num_charts = len(data['subset_srvs'])
    rows = math.ceil(num_charts / COLS)

    fig_w = CHART_WIDTH * COLS
    fig_h = TEXT_BLOCK_H + (CHART_HEIGHT * rows)

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)

    chart_block_h = CHART_HEIGHT * rows

    outer_gs = GridSpec(
        nrows=2,
        ncols=1,
        height_ratios=[TEXT_BLOCK_H, chart_block_h],
        hspace=0.1,
        figure=fig
    )

    top_gs = GridSpecFromSubplotSpec(
        nrows=3,
        ncols=1,
        subplot_spec=outer_gs[0],
        height_ratios=[0.5, 1.0, 0.5],
        hspace=0.0
    )

    # Title
    ax_title = fig.add_subplot(top_gs[0])
    ax_title.axis("off")
    ax_title.text(
        0.5, 0.5,
        "Validators Weights per Miner",
        ha="center", va="center",
        fontsize=14
    )

    # Description
    ax_para = fig.add_subplot(top_gs[1])
    ax_para.axis("off")
    ax_para.text(
        0.5, 0.5,
        textwrap.fill(description, width=140),
        ha="center", va="center",
        fontsize=11, wrap=False
    )

    # Legend placeholder
    ax_legend = fig.add_subplot(top_gs[2])
    ax_legend.axis("off")

    # Chart grid
    inner_gs = GridSpecFromSubplotSpec(
        nrows=rows,
        ncols=COLS,
        subplot_spec=outer_gs[1],
        wspace=0.3,
        hspace=0.4
    )

    styles = _get_validator_styles(data['subset_vals'])
    handles = []
    labels = []

    # Plot each server's chart
    for i_srv, srv in enumerate(data['subset_srvs']):
        r, c = divmod(i_srv, COLS)
        ax = fig.add_subplot(inner_gs[r, c])

        for i_val, val in enumerate(data['subset_vals']):
            series = data['data_cube'][i_srv][i_val]
            ls, mk, ms, mew = styles[val]
            line, = ax.plot(
                data['x'], series,
                linestyle=ls,
                marker=mk,
                markersize=ms,
                markeredgewidth=mew,
                linewidth=2,
                alpha=0.7
            )
            if i_srv == 0:
                handles.append(line)
                labels.append(data['hotkey_map'].get(val, val))

        ax.set_title(srv, fontsize=10)
        ax.grid(True)
        ax.set_ylim(0, 1.05)
        _set_default_xticks(ax, data['plot_epochs'])
        ax.set_xlabel("Epoch")
        if c == 0:
            ax.set_ylabel("Validator Weight")

    # Hide empty subplots
    total_slots = rows * COLS
    for idx in range(num_charts, total_slots):
        r, c = divmod(idx, COLS)
        ax_empty = fig.add_subplot(inner_gs[r, c])
        ax_empty.set_visible(False)

    # Add legend
    ncol = min(len(labels), 4)
    ax_legend.legend(
        handles, labels,
        loc="center",
        ncol=ncol,
        frameon=False,
        fontsize="small",
        handletextpad=0.3,
        columnspacing=0.5
    )

    fig.tight_layout(w_pad=0.3, h_pad=0.4)

    if to_base64:
        return _plot_to_base64(fig)

    plt.show()
    plt.close(fig)
    return None


def _plot_validator_server_weights_subplots_matplotlib(
    validators: list[str],
    weights_epochs: list[torch.Tensor],
    servers: list[str],
    num_epochs: int,
    case_name: str,
    to_base64: bool = False,
    epochs_padding: int = 0,
    **kwargs,
) -> str | None:
    """
    Plots validator weights in subplots (one subplot per server) over epochs using matplotlib.
    Each subplot shows lines for all validators, representing how much weight
    they allocate to that server from epoch 0..num_epochs-1.
    """
    data = _prepare_validator_server_weights_subplots_data(
        validators, weights_epochs, servers, num_epochs, epochs_padding
    )

    if data is None:
        return None

    x = data['x']
    data_matrix = data['data_matrix']

    fig, axes = plt.subplots(
        1,
        len(servers),
        figsize=(14, 5),
        sharex=True,
        sharey=True
    )

    if len(servers) == 1:
        axes = [axes]

    validator_styles = _get_validator_styles(validators)

    handles: list[plt.Artist] = []
    labels: list[str] = []

    for idx_s, server_name in enumerate(servers):
        ax = axes[idx_s]
        for idx_v, validator in enumerate(validators):
            y_values = data_matrix[idx_s][idx_v]
            linestyle, marker, markersize, markeredgewidth = validator_styles[validator]

            (line,) = ax.plot(
                x,
                y_values,
                alpha=0.7,
                marker=marker,
                markersize=markersize,
                markeredgewidth=markeredgewidth,
                linestyle=linestyle,
                linewidth=2,
                label=validator,
            )

            if idx_s == 0:
                handles.append(line)
                labels.append(validator)

        _set_default_xticks(ax, data['plot_epochs'])

        ax.set_xlabel("Epoch")
        if idx_s == 0:
            ax.set_ylabel("Validator Weight")
        ax.set_title(server_name)
        ax.set_ylim(0, 1.05)
        ax.grid(True)

    fig.suptitle(f"Validators Weights per Server\n{case_name}", fontsize=14)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(validators),
        bbox_to_anchor=(0.5, 0.02),
    )

    plt.tight_layout(rect=(0, 0.07, 1, 0.95))

    if to_base64:
        return _plot_to_base64(fig)

    plt.show()
    plt.close(fig)
    return None


def _plot_dividends_matplotlib(
    num_epochs: int,
    validators: list[str],
    dividends_per_validator: dict[str, list[float]],
    case_name: str,
    case: BaseCase,
    to_base64: bool = False,
) -> str | None:
    """
    Generates a matplotlib plot of dividends over epochs for a set of validators.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    data = _prepare_dividends_data(num_epochs, validators, dividends_per_validator, case)

    if data is None:
        return None

    plt.close("all")
    fig, ax_main = plt.subplots(figsize=(14, 6))

    validator_styles = _get_validator_styles(validators)

    for series_info in data['series_data']:
        validator = series_info['validator']
        linestyle, marker, markersize, markeredgewidth = validator_styles.get(
            validator, ("-", "o", 5, 1)
        )

        ax_main.plot(
            series_info['x_shifted'],
            series_info['data'],
            marker=marker,
            markeredgewidth=markeredgewidth,
            markersize=markersize,
            label=series_info['label'],
            alpha=0.7,
            linestyle=linestyle,
        )

    if data['num_epochs_calculated'] is not None:
        _set_default_xticks(ax_main, data['num_epochs_calculated'])

    ax_main.set_xlabel("Time (Epochs)")
    ax_main.set_ylim(bottom=0)
    ax_main.set_ylabel("Dividend per 1,000 Tao per Epoch")
    ax_main.set_title(case_name)
    ax_main.grid(True)
    ax_main.legend()

    ax_main.get_yaxis().set_major_formatter(ScalarFormatter(useMathText=True))
    ax_main.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))

    # Special case handling
    if case_name.startswith("Case 4"):
        ax_main.set_ylim(0, 0.042)

    plt.subplots_adjust(hspace=0.3)

    if to_base64:
        return _plot_to_base64(fig)
    else:
        plt.show()
        plt.close(fig)
        return None


def _plot_bonds_matplotlib(
    num_epochs: int,
    validators: list[str],
    servers: list[str],
    bonds_per_epoch: list[torch.Tensor],
    case_name: str,
    to_base64: bool = False,
    normalize: bool = False,
) -> str | None:
    """Generates a matplotlib plot of bonds per server for each validator."""

    # Process data
    bonds_data = _prepare_bonds_data(bonds_per_epoch, validators, servers, normalize=normalize)
    validator_styles = _get_validator_styles(validators)
    description, ylabel, title_suffix = _get_bonds_description_and_labels(normalize)

    x = list(range(num_epochs))

    fig, axes = plt.subplots(1, len(servers), figsize=(14, 5), sharex=True, sharey=True)
    if len(servers) == 1:
        axes = [axes]  # type: ignore

    handles: list[plt.Artist] = []
    labels: list[str] = []
    for idx_s, server in enumerate(servers):
        ax = axes[idx_s]
        for idx_v, validator in enumerate(validators):
            bonds = bonds_data[idx_s][idx_v]
            linestyle, marker, markersize, markeredgewidth = validator_styles[validator]

            (line,) = ax.plot(
                x,
                bonds,
                alpha=0.7,
                marker=marker,
                markersize=markersize,
                markeredgewidth=markeredgewidth,
                linestyle=linestyle,
                linewidth=2,
            )
            if idx_s == 0:
                handles.append(line)
                labels.append(validator)

        _set_default_xticks(ax, num_epochs)

        ax.set_xlabel("Epoch")
        if idx_s == 0:
            ax.set_ylabel(ylabel)
        ax.set_title(server)
        ax.grid(True)

        if normalize:
            ax.set_ylim(0, 1.05)

    fig.suptitle(
        f"Validators Bonds per Server{title_suffix}\n{case_name}",
        fontsize=14,
    )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(validators),
        bbox_to_anchor=(0.5, 0.02),
    )
    plt.tight_layout(rect=(0, 0.05, 0.98, 0.95))

    if to_base64:
        return _plot_to_base64(fig)

    plt.show()
    plt.close(fig)
    return None


def _validator_server_weights_special_handler_matplotlib(ax, data: dict, case_name: str):
    """Special matplotlib handler for validator server weights."""
    # Set custom y-axis ticks
    ax.set_yticks(data['y_tick_positions'])
    ax.set_yticklabels(data['y_tick_labels'])
    ax.set_ylim(data['y_range'])

    # Set title
    ax.set_title(f"Validators Weights to Servers \n{case_name}")


def _plot_validator_server_weights_matplotlib(
    validators: list[str],
    weights_epochs: list[torch.Tensor],
    servers: list[str],
    num_epochs: int,
    case_name: str,
    to_base64: bool = False,
) -> str | None:
    """Matplotlib implementation of validator server weights plot."""
    import matplotlib.pyplot as plt

    data = _prepare_validator_server_weights_data(
        validators, weights_epochs, servers, num_epochs
    )

    if data is None:
        return None

    # Create figure with adaptive height
    fig_height = data['fig_height_scale']
    fig, ax = plt.subplots(figsize=(14, fig_height))

    # Get validator styles
    validator_styles = _get_validator_styles(validators)

    # Plot each validator
    for series_info in data['series_data']:
        validator = series_info['validator']
        linestyle, marker, markersize, markeredgewidth = validator_styles[validator]

        ax.plot(
            data['x'],
            series_info['data'],
            label=series_info['label'],
            marker=marker,
            linestyle=linestyle,
            markersize=markersize,
            markeredgewidth=markeredgewidth,
            linewidth=2,
        )

    # Apply special formatting
    _validator_server_weights_special_handler_matplotlib(ax, data, case_name)

    # Standard formatting
    _set_default_xticks(ax, num_epochs)
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True)

    if to_base64:
        return _plot_to_base64(fig)
    plt.show()
    plt.close(fig)
    return None


# TODO: not refactored to plotly - only used for static raport right now
#       maybe we can clean it up :)
def _plot_relative_dividends_comparisson(
    validators_relative_dividends_normal: dict[str, list[float]],
    validators_relative_dividends_shifted: dict[str, list[float]],
    case: BaseCase,
    num_epochs: int,
    epochs_padding: int = 0,
    to_base64: bool = False,
    use_stakes: bool = False
) -> str | None:
    """
    Plots a comparison of dividends for each validator.
    The element-wise differences are plotted (shifted - normal), and the legend shows
    the difference in the means (displayed as a percentage).

    If 'use_stakes' is True and the case provides a stakes_dataframe property,
    then each difference is divided by the normalized stake for that validator at that epoch.
    The mean is recomputed from the newly calculated differences.

    The first `epochs_padding` records are omitted from the plot.
    """
    plt.close("all")
    # Adjust the number of epochs to be plotted.
    plot_epochs = num_epochs - epochs_padding
    if plot_epochs <= 0:
        logger.warning("Epochs padding is too large relative to number of total epochs. Nothing to plot.")
        return None

    fig, ax = plt.subplots(figsize=(14 * 2, 6 * 2))

    if not validators_relative_dividends_normal or not validators_relative_dividends_shifted:
        logger.warning("No validator data to plot.")
        return None

    all_validators = list(validators_relative_dividends_normal.keys())

    # Use the case's top validators if available; otherwise, plot all.
    top_vals = getattr(case, "requested_validators", [])
    if top_vals:
        plot_validator_names = top_vals.copy()
    else:
        plot_validator_names = all_validators.copy()

    # Ensure that the base (shifted) validator is included.
    base_validator = getattr(case, "base_validator", None)
    if base_validator and base_validator not in plot_validator_names:
        plot_validator_names.append(base_validator)

    x = np.arange(plot_epochs)
    validator_styles = _get_validator_styles(all_validators)

    # Retrieve stakes DataFrame if stakes normalization is requested.
    if use_stakes and hasattr(case, "stakes_dataframe"):
        df_stakes = case.stakes_dataframe
    else:
        df_stakes = None

    for idx, validator in enumerate(plot_validator_names):
        # Retrieve dividend series from both dictionaries.
        normal_dividends = validators_relative_dividends_normal.get(validator, [])
        shifted_dividends = validators_relative_dividends_shifted.get(validator, [])

        # Skip plotting if one of the series is missing or not long enough.
        if not normal_dividends or not shifted_dividends:
            continue
        if len(normal_dividends) <= epochs_padding or len(shifted_dividends) <= epochs_padding:
            continue

        # Slice off the first epochs_padding records.
        normal_dividends = normal_dividends[epochs_padding:]
        shifted_dividends = shifted_dividends[epochs_padding:]

        # Replace missing values (None) with np.nan.
        normal_dividends = np.array(
            [d if d is not None else np.nan for d in normal_dividends],
            dtype=float,
        )
        shifted_dividends = np.array(
            [d if d is not None else np.nan for d in shifted_dividends],
            dtype=float,
        )

        relative_diff = shifted_dividends - normal_dividends

        if df_stakes is not None and validator in df_stakes.columns:
            stakes_series = df_stakes[validator].to_numpy()
            # Ensure stakes series is sliced to match the dividends.
            if len(stakes_series) > epochs_padding:
                stakes_series = stakes_series[epochs_padding:]
            else:
                stakes_series = np.full_like(relative_diff, np.nan)
            with np.errstate(divide='ignore', invalid='ignore'):
                relative_diff = np.where(stakes_series != 0, relative_diff / stakes_series, np.nan)
            mean_difference = _compute_mean(relative_diff) * 100
        else:
            normal_mean = _compute_mean(normal_dividends)
            shifted_mean = _compute_mean(shifted_dividends)
            mean_difference = (shifted_mean - normal_mean) * 100

        delta = 0.05
        x_shifted = x + idx * delta

        sign_str = f"{mean_difference:+.5f}%"
        label = f"{validator}: mean difference/stake = {sign_str}" if use_stakes else f"{validator}: mean difference = {sign_str}"

        linestyle, marker, markersize, markeredgewidth = validator_styles.get(
            validator, ("-", "o", 5, 1)
        )

        ax.plot(
            x_shifted,
            relative_diff,
            label=label,
            alpha=0.7,
            marker=marker,
            markeredgewidth=markeredgewidth,
            markersize=markersize,
            linestyle=linestyle,
        )

    ax.axhline(y=0, color="black", linewidth=1, linestyle="--", alpha=0.7)

    _set_default_xticks(ax, plot_epochs)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Absolute Difference")
    ax.set_title("Comparison (shifted - normal) scaled by stake" if use_stakes else "Comparison (shifted - normal)")
    ax.grid(True)

    legend = ax.legend()
    for text in legend.get_texts():
        if text.get_text().startswith(case.shift_validator_hotkey):
            text.set_fontweight('bold')

    def to_percent(y, _):
        return f"{y * 100:.1f}%"
    ax.yaxis.set_major_formatter(FuncFormatter(to_percent))

    plt.subplots_adjust(hspace=0.3)

    if to_base64:
        return _plot_to_base64(fig)
    else:
        plt.show()
        plt.close(fig)
        return None


def _plot_incentives_matplotlib(
    servers: list[str],
    server_incentives_per_epoch: list[torch.Tensor],
    num_epochs: int,
    case_name: str,
    case: BaseCase = None,
    to_base64: bool = False,
) -> str | None:
    """Generates a matplotlib plot of server incentives over epochs."""

    data = _prepare_incentives_data(servers, server_incentives_per_epoch, num_epochs, case)

    if data is None:
        return None

    x = np.arange(num_epochs)
    fig, ax = plt.subplots(figsize=(14, 3))

    for series_info in data['series_data']:
        ax.plot(x, series_info['data'], label=series_info['label'])

    _set_default_xticks(ax, num_epochs)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Server Incentive")
    ax.set_title(f"Server Incentives\n{case_name}")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True)

    if to_base64:
        return _plot_to_base64(fig)
    plt.show()
    plt.close(fig)
    return None
