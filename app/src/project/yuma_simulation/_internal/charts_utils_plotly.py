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


def _compute_mean(dividends: np.ndarray) -> float:
    """Computes the mean over valid epochs where the validator is present."""
    if np.all(np.isnan(dividends)):
        return 0.0
    return np.nanmean(dividends)


def _get_validator_styles(
    validators: list[str],
) -> dict[str, tuple[str, str, int, int]]:
    combined_styles = [("-", "+", 12, 2), ("--", "x", 12, 1), (":", "o", 4, 1)]
    return {
        validator: combined_styles[idx % len(combined_styles)]
        for idx, validator in enumerate(validators)
    }


def _plot_relative_dividends(
    validators_relative_dividends: dict[str, list[float]],
    case_name: str,
    case: BaseCase,
    num_epochs: int,
    epochs_padding: int = 0,
    **kwargs,
) -> str:

    plot_epochs = num_epochs - epochs_padding
    if plot_epochs <= 0 or not validators_relative_dividends:
        return '<div style="color:red;">Nothing to plot (padding/empty data).</div>'

    # Generate unique ID for this chart
    chart_id = f"chart-{uuid.uuid4().hex[:8]}"

    # Create figure
    fig = go.Figure()

    # Prepare data exactly like matplotlib version
    all_validators = list(validators_relative_dividends.keys())
    top_vals = getattr(case, "requested_validators", []) or all_validators.copy()
    if case.base_validator not in top_vals:
        top_vals.append(case.base_validator)

    x = list(range(plot_epochs))
    validator_styles = _get_validator_styles(all_validators)

    # Define clean colors and line styles
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Plot each validator with clean styling
    for idx, validator in enumerate(top_vals):
        series = validators_relative_dividends.get(validator, [])
        if len(series) <= epochs_padding:
            continue

        arr = np.array([d if d is not None else np.nan for d in series[epochs_padding:]], dtype=float)

        # Apply the same x-shifting as matplotlib version
        x_shifted = [x_val + idx * 0.05 for x_val in x]

        mean_pct = _compute_mean(arr) * 100
        label = f"{case.hotkey_label_map.get(validator, validator[:10] + '...')}: total = {mean_pct:+.5f}%"

        # Get matplotlib styles
        ls, mk, ms, mew = validator_styles.get(validator, ("-", "o", 5, 1))

        # Convert matplotlib line styles to Plotly
        line_dash = {
            "-": "solid",
            "--": "dash",
            ":": "dot",
            "-.": "dashdot"
        }.get(ls, "solid")

        # Get color
        color = colors[idx % len(colors)]

        # Determine if this is the shift validator
        line_width = 2 if validator == getattr(case, 'shift_validator_hotkey', None) else 1.5

        fig.add_trace(
            go.Scatter(
                x=x_shifted,
                y=arr * 100,
                mode='lines+markers',
                name=label,
                line=dict(
                    dash=line_dash,
                    width=line_width,
                    color=color
                ),
                marker=dict(
                    symbol='circle',
                    size=4,
                    color=color,
                    line=dict(width=0.5, color=color)
                ),
                opacity=0.8
            )
        )

    # Update layout without title and with proper legend
    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
            itemsizing="trace",
            itemwidth=30,
            traceorder="normal"
        ),
        margin=dict(l=60, r=40, t=20, b=100),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Update axes
    fig.update_xaxes(
        title_text="Epoch",
        title_font_size=12,
        showgrid=True,
        gridcolor='rgba(128,128,128,0.2)',
        gridwidth=0.5,
        zeroline=False
    )

    fig.update_yaxes(
        title_text="Relative Dividend (%)",
        title_font_size=12,
        showgrid=True,
        gridcolor='rgba(128,128,128,0.2)',
        gridwidth=0.5,
        tickformat=".1f",
        ticksuffix="%",
        zeroline=False
    )

    # Convert to JSON config
    chart_config = {
        "data": json.loads(json.dumps(fig.data, cls=plotly.utils.PlotlyJSONEncoder)),
        "layout": json.loads(json.dumps(fig.layout, cls=plotly.utils.PlotlyJSONEncoder)),
        "config": {
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ['pan2d', 'lasso2d', 'select2d'],
            "responsive": True
        }
    }

    # Prepare description text
    description = (
        "“Validator Relative Dividends” is a performance metric that measures the deviation "
        "between a validator's actual dividends and the hypothetical zero-sum dividends they "
        "would have earned purely in proportion to their stake. This difference highlights "
        "whether a validator has over- or under-performed relative to the stake-weighted "
        "zero-sum baseline across the entire network."
    )

    html = f"""
    <!-- Title Section -->
    <div class="mb-4">
        <h5 class="mb-2">
            Validators Relative Dividends
        </h5>
        <h6 class="text-muted mb-3">
            {case_name}
        </h6>
        <p class="text-muted mb-0" style="font-size: 0.95rem; line-height: 1.5;">
            {description}
        </p>
    </div>

    <!-- Formula Badge -->
    <div class="mb-4">
        <div class="alert alert-light border d-inline-block py-2 px-3 mb-0">
            <small class="text-dark">
                <strong>Relative Dividend =</strong>
                <code class="ms-1">
                    (Validator's Dividends / Σ Dividends) - (Validator's Stake / Σ Stake)
                </code>
            </small>
        </div>
    </div>

    <!-- Chart Container -->
    <div id="{chart_id}" style="width:100%; height:500px;"></div>

    <script type="application/json" data-chart-config="{chart_id}">
    {json.dumps(chart_config, indent=2)}
    </script>
    <script type="text/javascript" data-chart-render="{chart_id}">
    (function() {{
        if (typeof Plotly === 'undefined') {{
            console.error('Plotly is not loaded');
            return;
        }}

        try {{
            const configScript = document.querySelector('script[data-chart-config="{chart_id}"]');
            const chartConfig = JSON.parse(configScript.textContent);

            Plotly.newPlot('{chart_id}', chartConfig.data, chartConfig.layout, chartConfig.config);

        }} catch (error) {{
            console.error('Error rendering chart {chart_id}:', error);
            document.getElementById('{chart_id}').innerHTML =
                '<div class="alert alert-danger text-center">Error rendering chart: ' + error.message + '</div>';
        }}
    }})();
    </script>
    """

    return html


def _plot_validator_server_weights_subplots_dynamic(
    case: MetagraphCase,
    default_miners: list[str],
    case_name: str,
    epochs_padding: int = 0,
    **kwargs,
) -> str | None:
    """
    Dynamic version for metagraph-based weights, skipping the first `epochs_padding` epochs
    """
    total_epochs = case.num_epochs
    plot_epochs = total_epochs - epochs_padding
    if plot_epochs <= 0:
        return '<div class="alert alert-warning">Nothing to plot (padding >= total_epochs).</div>'

    subset_vals = case.requested_validators or case.validators_epochs[0]
    subset_srvs = case.selected_servers or default_miners
    hotkey_map = case.hotkey_label_map
    weights_epochs = case.weights_epochs

    # Build data cube
    data_cube: list[list[list[float]]] = []
    for srv in subset_srvs:
        per_val = []
        for val in subset_vals:
            series = []
            for e in range(epochs_padding, total_epochs):
                ve, se, W = case.validators_epochs[e], case.servers[e], weights_epochs[e]
                if (val in ve) and (srv in se):
                    r, c = ve.index(val), se.index(srv)
                    series.append(float(W[r, c].item()))
                else:
                    series.append(np.nan)
            per_val.append(series)
        data_cube.append(per_val)

    # Generate unique ID for this chart
    chart_id = f"chart-{uuid.uuid4().hex[:8]}"

    # Setup subplot grid
    COLS = 2
    num_charts = len(subset_srvs)
    rows = math.ceil(num_charts / COLS)

    # Create subplots (keep this part)
    fig = make_subplots(
        rows=rows,
        cols=COLS,
        subplot_titles=[srv for srv in subset_srvs],
        vertical_spacing=0.08,
        horizontal_spacing=0.1,
        specs=[[{"secondary_y": False} for _ in range(COLS)] for _ in range(rows)]
    )

    # Define colors and styles upfront
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    styles = _get_validator_styles(subset_vals)
    x = list(range(plot_epochs))

    trace_objs = []   # raw dicts instead of go.Scatter
    row_idx    = []
    col_idx    = []

    for i_srv, srv in enumerate(subset_srvs):
        row = (i_srv // COLS) + 1
        col = (i_srv % COLS) + 1

        for i_val, val in enumerate(subset_vals):
            series = data_cube[i_srv][i_val]

            # Get matplotlib styles
            ls, mk, ms, mew = styles[val]

            # Convert matplotlib line styles to Plotly
            line_dash = {
                "-": "solid",
                "--": "dash",
                ":": "dot",
                "-.": "dashdot"
            }.get(ls, "solid")

            # Get color
            color = colors[i_val % len(colors)]

            # Only show legend for first subplot
            show_legend = (i_srv == 0)
            legend_name = hotkey_map.get(val, val) if show_legend or True else None
            line_width = 2 if val == getattr(case, 'shift_validator_hotkey', None) else 1.5

            trace_objs.append({
                "type": "scattergl",    # or "scatter"
                "x": x,
                "y": series,
                "mode": "lines+markers",
                "name": legend_name,
                "line": {
                    "dash": line_dash,
                    "width": line_width,
                    "color": color
                },
                "marker": {
                    "symbol": "circle",
                    "size": 3,
                    "color": color,
                    "line": {"width": 0.5, "color": color}
                },
                "opacity": 0.8,
                "showlegend": show_legend,
                "legendgroup": val
            })
            row_idx.append(row)         # 1-based subplot indices
            col_idx.append(col)

    fig.add_traces(trace_objs, rows=row_idx, cols=col_idx)

    # Build axis updates as dictionaries for batch update
    xaxis_updates = {}
    yaxis_updates = {}

    for i in range(1, rows + 1):
        for j in range(1, COLS + 1):
            subplot_num = (i-1)*COLS + j
            if subplot_num > num_charts:
                continue

            # Prepare axis update dictionaries
            xaxis_key = f'xaxis{subplot_num}' if subplot_num > 1 else 'xaxis'
            yaxis_key = f'yaxis{subplot_num}' if subplot_num > 1 else 'yaxis'

            xaxis_updates[xaxis_key] = {
                'title': "Epoch",
                'showgrid': True,
                'gridcolor': 'rgba(128,128,128,0.2)',
                'gridwidth': 0.5
            }

            yaxis_updates[yaxis_key] = {
                'title': "Validator Weight" if j == 1 else "",
                'range': [0, 1.05],
                'showgrid': True,
                'gridcolor': 'rgba(128,128,128,0.2)',
                'gridwidth': 0.5
            }

    # Update layout in one call with all properties
    fig.update_layout({
        'height': 400 + (rows * 250),
        'showlegend': True,
        'legend': {
            'orientation': "h",
            'yanchor': "top",
            'y': -0.05,
            'xanchor': "center",
            'x': 0.5,
            'font': {'size': 10},
            'bgcolor': "rgba(255,255,255,0.9)",
            'bordercolor': "rgba(0,0,0,0.1)",
            'borderwidth': 1,
            'itemsizing': "trace"
        },
        'margin': {'l': 60, 'r': 40, 't': 60, 'b': 80},
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        **xaxis_updates,
        **yaxis_updates,
    })

    # Convert to JSON config
    chart_config = {
        "data": json.loads(json.dumps(fig.data, cls=plotly.utils.PlotlyJSONEncoder)),
        "layout": json.loads(json.dumps(fig.layout, cls=plotly.utils.PlotlyJSONEncoder)),
        "config": {
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ['pan2d', 'lasso2d', 'select2d'],
            "responsive": True
        }
    }

    # Description
    description = (
        "'Validators Weights per Miner' is a visualization that shows how "
        "validators allocate their stake to different miners over time. "
        "Each line represents a validator's weight on a specific miner."
    )

    html = f"""
    <!-- Title Section -->
    <div class="mb-4">
        <h5 class="mb-2">
            Validators Weights per Miner
        </h5>
        <h6 class="text-muted mb-3">
            {case_name}
        </h6>
        <p class="text-muted mb-0" style="font-size: 0.95rem; line-height: 1.5;">
            {description}
        </p>
    </div>

    <!-- Chart Container -->
    <div id="{chart_id}" style="width:100%; height:{400 + (rows * 250)}px;"></div>

    <script type="application/json" data-chart-config="{chart_id}">
    {json.dumps(chart_config, indent=2)}
    </script>
    <script type="text/javascript" data-chart-render="{chart_id}">
    (function() {{
        if (typeof Plotly === 'undefined') {{
            console.error('Plotly is not loaded');
            return;
        }}

        try {{
            const configScript = document.querySelector('script[data-chart-config="{chart_id}"]');
            const chartConfig = JSON.parse(configScript.textContent);

            Plotly.newPlot('{chart_id}', chartConfig.data, chartConfig.layout, chartConfig.config);

        }} catch (error) {{
            console.error('Error rendering chart {chart_id}:', error);
            document.getElementById('{chart_id}').innerHTML =
                '<div class="alert alert-danger text-center">Error rendering chart: ' + error.message + '</div>';
        }}
    }})();
    </script>
    """

    return html

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


def _plot_bonds_metagraph_dynamic(
    case: MetagraphCase,
    default_miners: list[str],
    bonds_per_epoch: list[torch.Tensor],
    case_name: str,
    normalize: bool = False,
    legend_validators: list[str] | None = None,
    epochs_padding: int = 0,
    **kwargs,
) -> str | None:

    num_epochs = case.num_epochs
    plot_epochs = num_epochs - epochs_padding
    if plot_epochs <= 0:
        logger.warning("Nothing to plot (padding >= total_epochs).")
        return '<div class="alert alert-warning">Nothing to plot (padding >= total_epochs).</div>'

    validators_epochs = case.validators_epochs
    miners_epochs = case.servers
    selected_validators = case.requested_validators

    bonds_data = _prepare_bond_data_dynamic(
        bonds_per_epoch, validators_epochs, miners_epochs,
        normalize=normalize,
    )

    subset_v = selected_validators or validators_epochs[0]
    subset_m = case.selected_servers or default_miners

    miner_keys = miners_epochs[0]
    validator_keys: list[str] = []
    for epoch in validators_epochs:
        for v in epoch:
            if v not in validator_keys:
                validator_keys.append(v)

    #TODO(handle this better)
    m_idx = []
    for m in subset_m:
        mi = None
        for me in miners_epochs:
            try:
                mi = me.index(m)
            except ValueError:
                pass
            else:
                break
        if mi is None:
            raise RuntimeError('AAAAA')
        else:
            m_idx.append(mi)
    # m_idx = [miner_keys.index(m)     for m in subset_m]
    v_idx = []
    for v in subset_v:
        vi = None
        for ve in validators_epochs:
            try:
                vi = ve.index(v)
            except ValueError:
                pass
            else:
                break
        if vi is None:
            raise RuntimeError('AAAAA')
        else:
            v_idx.append(vi)
    # v_idx = [validator_keys.index(v) for v in subset_v]

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

    # Generate unique ID for this chart
    chart_id = f"chart-{uuid.uuid4().hex[:8]}"

    # Setup subplot grid
    COLS = 2
    num_charts = len(subset_m)
    rows = math.ceil(num_charts / COLS)

    # Create subplots
    fig = make_subplots(
        rows=rows,
        cols=COLS,
        subplot_titles=[miner for miner in subset_m],
        vertical_spacing=0.08,
        horizontal_spacing=0.1,
        specs=[[{"secondary_y": False} for _ in range(COLS)] for _ in range(rows)]
    )

    # Define colors and styles upfront
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    styles = _get_validator_styles(validator_keys)
    x = list(range(plot_epochs))

    # Pre-build traces as raw dicts for faster creation
    trace_objs = []   # raw dicts instead of go.Scatter
    row_idx = []
    col_idx = []

    for i_miner, miner in enumerate(subset_m):
        row = (i_miner // COLS) + 1
        col = (i_miner % COLS) + 1

        for j, val in enumerate(subset_v):
            # Get matplotlib styles
            ls, mk, ms, mew = styles[val]

            # Convert matplotlib line styles to Plotly
            line_dash = {
                "-": "solid",
                "--": "dash",
                ":": "dot",
                "-.": "dashdot"
            }.get(ls, "solid")

            # Get color
            color = colors[j % len(colors)]

            # Determine if this validator should be in legend
            should_show_legend = (
                i_miner == 0 and
                (legend_validators or subset_v) and
                (val in (legend_validators or subset_v))
            )

            legend_name = case.hotkey_label_map.get(val, val) if should_show_legend or True else None
            line_width = 2 if val == getattr(case, 'shift_validator_hotkey', None) else 1.5

            # Create trace as raw dict for faster performance
            trace_objs.append({
                "type": "scattergl",  # Use scattergl for better performance with many points
                "x": x,
                "y": plot_data[i_miner][j],
                "mode": "lines+markers",
                "name": legend_name,
                "line": {
                    "dash": line_dash,
                    "width": line_width,
                    "color": color
                },
                "marker": {
                    "symbol": "circle",
                    "size": 3,
                    "color": color,
                    "line": {"width": 0.5, "color": color}
                },
                "opacity": 0.8,
                "showlegend": should_show_legend,
                "legendgroup": val
            })
            row_idx.append(row)
            col_idx.append(col)

    # Add all traces at once - much faster than individual add_trace calls
    fig.add_traces(trace_objs, rows=row_idx, cols=col_idx)

    # Determine y-axis settings based on normalize flag
    if normalize:
        y_range = [0, 1.05]
        ylabel = "Bond Ratio"
        title_suffix = " (Normalized)"
        description = (
            "This plot shows each miner's normalized bond ratio from each validator over time. "
            "At every epoch, each miner's incoming bonds have been scaled so that their total across "
            "all validators equals 1."
        )
    else:
        y_range = [0, None]
        ylabel = "Bond Value"
        title_suffix = ""
        description = (
            "This plot shows each validator's absolute bond value to each miner over time. "
            "At every epoch, the raw bond tensor is used, in the native units of a given simulation version."
        )

    # Build axis updates as dictionaries for batch update
    xaxis_updates = {}
    yaxis_updates = {}

    for i in range(1, rows + 1):
        for j in range(1, COLS + 1):
            subplot_num = (i-1)*COLS + j
            if subplot_num > num_charts:
                continue

            # Prepare axis update dictionaries
            xaxis_key = f'xaxis{subplot_num}' if subplot_num > 1 else 'xaxis'
            yaxis_key = f'yaxis{subplot_num}' if subplot_num > 1 else 'yaxis'

            xaxis_updates[xaxis_key] = {
                'title': "Epoch",
                'showgrid': True,
                'gridcolor': 'rgba(128,128,128,0.2)',
                'gridwidth': 0.5
            }

            yaxis_updates[yaxis_key] = {
                'title': ylabel if j == 1 else "",
                'range': y_range,
                'showgrid': True,
                'gridcolor': 'rgba(128,128,128,0.2)',
                'gridwidth': 0.5
            }
    # Update layout in one call with all properties
    fig.update_layout({
        'height': 400 + (rows * 250),
        'showlegend': True,
        'legend': {
            'orientation': "h",
            'yanchor': "top",
            'y': -0.05,
            'xanchor': "center",
            'x': 0.5,
            'font': {'size': 10},
            'bgcolor': "rgba(255,255,255,0.9)",
            'bordercolor': "rgba(0,0,0,0.1)",
            'borderwidth': 1,
            'itemsizing': "trace"
        },
        'margin': {'l': 60, 'r': 40, 't': 60, 'b': 80},
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        **xaxis_updates,
        **yaxis_updates,
    })

    # Convert to JSON config
    chart_config = {
        "data": json.loads(json.dumps(fig.data, cls=plotly.utils.PlotlyJSONEncoder)),
        "layout": json.loads(json.dumps(fig.layout, cls=plotly.utils.PlotlyJSONEncoder)),
        "config": {
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ['pan2d', 'lasso2d', 'select2d'],
            "responsive": True
        }
    }

    html = f"""
    <!-- Title Section -->
    <div class="mb-4">
        <h5 class="mb-2">
            Validators Bonds per Miner{title_suffix}
        </h5>
        <h6 class="text-muted mb-3">
            {case_name}
        </h6>
        <p class="text-muted mb-0" style="font-size: 0.95rem; line-height: 1.5;">
            {description}
        </p>
    </div>

    <!-- Chart Container -->
    <div id="{chart_id}" style="width:100%; height:{400 + (rows * 250)}px;"></div>

    <script type="application/json" data-chart-config="{chart_id}">
    {json.dumps(chart_config, indent=2)}
    </script>
    <script type="text/javascript" data-chart-render="{chart_id}">
    (function() {{
        if (typeof Plotly === 'undefined') {{
            console.error('Plotly is not loaded');
            return;
        }}

        try {{
            const configScript = document.querySelector('script[data-chart-config="{chart_id}"]');
            const chartConfig = JSON.parse(configScript.textContent);

            Plotly.newPlot('{chart_id}', chartConfig.data, chartConfig.layout, chartConfig.config);

        }} catch (error) {{
            console.error('Error rendering chart {chart_id}:', error);
            document.getElementById('{chart_id}').innerHTML =
                '<div class="alert alert-danger text-center">Error rendering chart: ' + error.message + '</div>';
        }}
    }})();
    </script>
    """

    return html
