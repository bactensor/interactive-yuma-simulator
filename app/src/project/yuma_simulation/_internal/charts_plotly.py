import math
import json
import uuid
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils

import torch

from project.yuma_simulation._internal.cases import BaseCase, MetagraphCase
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


class ChartType(Enum):
    LINE = "line"
    SUBPLOT_GRID = "subplot_grid"


@dataclass
class ChartConfig:
    """Enhanced chart configuration with smart defaults"""
    height: int = 500
    show_legend: bool = True
    grid_legend: bool = False
    legend_horizontal: bool = True
    colors: List[str] = None
    line_styles: List[str] = None
    grid_color: str = 'rgba(128,128,128,0.2)'
    grid_width: float = 0.5
    background_color: str = 'rgba(255,255,255,0)'
    paper_color: str = 'rgba(255,255,255,0)'
    # Enhanced y-axis configuration
    y_axis_format: str = "auto"  # "auto", "scientific", "fixed", "percentage"
    y_axis_range: Tuple[Optional[float], Optional[float]] = (None, None)
    y_axis_zero_line: bool = False
    custom_y_axis_config: Optional[Dict[str, Any]] = None
    # X-axis shifting for overlapping points
    enable_x_shift: bool = False
    x_shift_delta: float = 0.05

    def __post_init__(self):
        if self.colors is None:
            self.colors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]
        if self.line_styles is None:
            self.line_styles = ["solid", "dash", "dot", "dashdot"]


@dataclass
class SeriesStyle:
    """Style configuration for a data series"""
    color: str
    line_dash: str = "solid"
    line_width: float = 1.5
    marker_size: int = 4
    opacity: float = 0.8
    show_in_legend: bool = True
    legend_group: Optional[str] = None
    legend_font_weight: Optional[str] = None  # "bold", "normal", "bold+underline"


class PlotlyChartBuilder:
    """Chart builder with sensible defaults and common patterns"""

    def __init__(self, config: ChartConfig = None):
        self.config = config or ChartConfig()
        self.chart_id = f"chart-{uuid.uuid4().hex[:8]}"
        self._validator_styles_cache = {}

    def _get_validator_styles(self, validators: List[str]) -> Dict[str, Tuple[str, str, int, int]]:
        """Get consistent styles for validators with caching"""
        cache_key = tuple(sorted(validators))
        if cache_key not in self._validator_styles_cache:
            combined_styles = [("-", "+", 12, 2), ("--", "x", 12, 1), (":", "o", 4, 1)]
            self._validator_styles_cache[cache_key] = {
                validator: combined_styles[idx % len(combined_styles)]
                for idx, validator in enumerate(validators)
            }
        return self._validator_styles_cache[cache_key]

    def _matplotlib_to_plotly_dash(self, matplotlib_style: str) -> str:
        """Convert matplotlib line styles to Plotly"""
        return {
            "-": "solid",
            "--": "dash",
            ":": "dot",
            "-.": "dashdot"
        }.get(matplotlib_style, "solid")

    def _create_series_style(self,
                           validator: str,
                           idx: int,
                           case: BaseCase = None,
                           emphasized_validator: str = None) -> SeriesStyle:
        """Create consistent series styling"""
        color = self.config.colors[idx % len(self.config.colors)]
        line_dash = self.config.line_styles[idx % len(self.config.line_styles)]

        line_width = 1.5
        marker_size = 4
        legend_font_weight = "normal"

        if emphasized_validator:
            # Emphasize specific validator (e.g., shift validator)
            line_width = 4.0 if validator == emphasized_validator else 0.5
            marker_size = 4 if validator == emphasized_validator else 3
            legend_font_weight = "bold+underline" if validator == emphasized_validator else "normal"

        return SeriesStyle(
            color=color,
            line_dash=line_dash,
            line_width=line_width,
            legend_group=validator,
            marker_size=marker_size,
            legend_font_weight=legend_font_weight,
        )

    def _create_base_layout(self,
                          title: str = None,
                          x_title: str = "Epoch",
                          y_title: str = None,
                          height: int = None) -> Dict[str, Any]:
        """Create base layout with enhanced y-axis configuration"""
        height = height or self.config.height

        layout = {
            'height': height,
            'showlegend': self.config.show_legend,
            'margin': {
                'l': 60,
                'r': 40,
                't': 60 if title else 40,
                'b': 60,
            },
            'plot_bgcolor': self.config.background_color,
            'paper_bgcolor': self.config.paper_color,
        }

        if self.config.show_legend:
            layout['legend'] = {
                'orientation': "h" if self.config.legend_horizontal else "v",
                'yanchor': "bottom",
                'y': -layout["margin"]['b'] / height if self.config.legend_horizontal else 1,
                'yref': 'container',
                'xanchor': "center" if self.config.legend_horizontal else "left",
                'x': 0.5 if self.config.legend_horizontal else 1.02,
                'font': {'size': 11},
                'bgcolor': "rgba(255,255,255,0.95)",
                'bordercolor': "rgba(0,0,0,0.1)",
                'borderwidth': 1,
                'itemsizing': "trace",
            }
            if self.config.grid_legend:
                layout['legend'].update({
                    'itemwidth': 30,
                    'traceorder': "normal",
                })

        # Enhanced axis styling
        axis_style = {
            'showgrid': True,
            'gridcolor': self.config.grid_color,
            'gridwidth': self.config.grid_width,
            'zeroline': self.config.y_axis_zero_line
        }

        layout['xaxis'] = {**axis_style, 'title': x_title}
        layout['yaxis'] = {**axis_style, 'title': y_title}

        # Configure y-axis formatting
        if self.config.y_axis_format == "scientific":
            layout['yaxis'].update({
                'tickformat': '.2e',
                'exponentformat': 'e',
                'showexponent': 'all'
            })
        elif self.config.y_axis_format == "percentage":
            layout['yaxis'].update({
                'tickformat': '.1%'
            })

        # Configure y-axis range
        if self.config.y_axis_range != (None, None):
            y_min, y_max = self.config.y_axis_range
            if y_min is None:
                layout['yaxis']['autorange']= "min"
                layout['yaxis']['autorangeoptions']= {"include": y_max}
            elif y_max is None:
                layout['yaxis']['autorange']= "max"
                layout['yaxis']['autorangeoptions']= {"include": y_min}
            else:
                layout['yaxis']['range'] = [y_min, y_max]

        # Apply custom y-axis configuration
        if self.config.custom_y_axis_config:
            layout['yaxis'].update(self.config.custom_y_axis_config)

        return layout

    def _create_trace_dict(self,
                          x_data: List,
                          y_data: List,
                          name: str,
                          style: SeriesStyle) -> Dict[str, Any]:
        """Create a plotly trace dict with consistent styling"""
        if style.legend_font_weight == "bold":
            name = f"<b>{name}</b>"
        elif style.legend_font_weight == "bold+underline":
            name = f"<b><u>{name}</u></b>"
        return {
            "type": "scatter",
            "x": x_data,
            "y": y_data,
            "mode": "lines+markers",
            "name": name,
            "line": {
                "dash": style.line_dash,
                "width": style.line_width,
                "color": style.color
            },
            "marker": {
                "symbol": "circle",
                "size": style.marker_size,
                "color": style.color,
                "line": {"width": style.line_width, "color": style.color}
            },
            "opacity": style.opacity,
            "showlegend": style.show_in_legend,
            "legendgroup": style.legend_group
        }


    def _generate_html(self,
                      fig: go.Figure,
                      title: str,
                      subtitle: str = None,
                      description: str = None,
                      formula: str = None) -> str:
        """Generate complete HTML with consistent structure"""

        # Convert to JSON config
        chart_config = {
            "data": json.loads(json.dumps(fig.data, cls=plotly.utils.PlotlyJSONEncoder)),
            "layout": json.loads(json.dumps(fig.layout, cls=plotly.utils.PlotlyJSONEncoder)),
            "config": {
                "displayModeBar": 'hover',
                "displaylogo": False,
                "modeBarButtons": [
                    ["autoScale2d", 'resetViews', 'toImage'],
                ],
                "responsive": True
            }
        }

        # Build HTML sections
        title_section = f"""
        <div>
            {f'<h6 class="mb-2">{title}</h6>' if title else ''}
            {f'<h7 class="text-muted mb-3">{subtitle}</h7>' if subtitle else ''}
            {f'<p class="text-muted mb-0 plotly-chart-html-description" style="font-size: 0.95rem; line-height: 1.5;">{description}</p>' if description else ''}
        </div>
        """

        formula_section = ""
        if formula:
            formula_section = f"""
            <div class="plotly-chart-html-description">
                <div class="alert alert-light border d-inline-block py-2 px-3 mb-0">
                    <small class="text-dark">
                        <pre class="m-0"><code>{formula}</code></pre>
                    </small>
                </div>
            </div>
            """

        chart_height = fig.layout.height or self.config.height

        return f"""
        {title_section}
        {formula_section}

        <div id="{self.chart_id}" style="width:100%; height:{chart_height}px;"></div>

        <script type="application/json" data-chart-config="{self.chart_id}">
        {json.dumps(chart_config, indent=2)}
        </script>
        <script type="text/javascript" data-chart-render="{self.chart_id}">
        (function() {{
            if (typeof Plotly === 'undefined') {{
                console.error('Plotly is not loaded');
                return;
            }}

            try {{
                const configScript = document.querySelector('script[data-chart-config="{self.chart_id}"]');
                const chartConfig = JSON.parse(configScript.textContent);

                Plotly.newPlot('{self.chart_id}', chartConfig.data, chartConfig.layout, chartConfig.config);

            }} catch (error) {{
                console.error('Error rendering chart {self.chart_id}:', error);
                document.getElementById('{self.chart_id}').innerHTML =
                    '<div class="alert alert-danger text-center">Error rendering chart: ' + error.message + '</div>';
            }}
        }})();
        </script>
        """


def _adapt_data_for_subplots(data: Dict[str, Any],
                           chart_titles_key: str,
                           series_keys_key: str,
                           data_matrix_key: str) -> Dict[str, Any]:
    """Common data adaptation for subplot grids with parametrized keys"""
    if not data:
        return None

    chart_titles = data[chart_titles_key]
    series_keys = data[series_keys_key]
    data_matrix = data[data_matrix_key]

    chart_data = []
    for i_chart in range(len(chart_titles)):
        series_list = []
        for j, series_key in enumerate(series_keys):
            series_list.append({
                'key': series_key,
                'y_data': data_matrix[i_chart][j]
            })
        chart_data.append({'series': series_list})

    return {
        'subset_titles': chart_titles,
        'chart_data': chart_data,
        'x': data['x'],
        'y_range': data.get('y_range', [None, None])
    }


class ChartFactory:
    """Factory for creating common chart types with minimal configuration"""

    @staticmethod
    def create_line_chart(data: Dict[str, Any],
                         case: BaseCase,
                         case_name: str,
                         title: str | None = None,
                         y_label: str = None,
                         description: str = None,
                         formula: str = None,
                         config: ChartConfig = None,
                         special_case_handler: callable = None,
                         series_key: str = 'validator') -> str:
        """Create a line chart with automatic styling and special case handling"""

        if data is None:
            return '<div style="color:red;">Nothing to plot (padding/empty data).</div>'

        builder = PlotlyChartBuilder(config)
        config = builder.config
        fig = go.Figure()

        emphasized_validator = getattr(case, 'shift_validator_hotkey', None)

        # Build all trace dicts first
        trace_dicts = []
        for series_info in data['series_data']:
            style = builder._create_series_style(
                series_info[series_key],
                series_info['idx'],
                case,
                emphasized_validator
            )

            # Handle x-axis shifting if enabled
            if config and config.enable_x_shift:
                x_data = series_info.get('x_shifted', series_info.get('x', data.get('x', [])))
            else:
                x_data = series_info.get('x', data.get('x', []))

            # Truncate label for display if needed
            display_label = series_info['label']
            if hasattr(case, 'hotkey_label_map'):
                validator_key = series_info['validator']
                original_name = case.hotkey_label_map.get(validator_key, validator_key)
                if len(original_name) > 25:
                    short_name = original_name[:22] + '...'
                    display_label = display_label.replace(original_name, short_name)

            trace_dict = builder._create_trace_dict(
                x_data,
                series_info.get('data', series_info.get('y_data', [])),
                display_label,
                style
            )
            trace_dicts.append(trace_dict)

        # Add all traces at once
        fig.add_traces(trace_dicts)

        # Apply layout
        layout = builder._create_base_layout(
            title=None,  # explicitely don't draw title - we use html title
            y_title=y_label,
        )

        # Apply special case handler if provided
        if special_case_handler:
            layout = special_case_handler(layout, case_name, data)

        fig.update_layout(layout)

        return builder._generate_html(
            fig, title, case_name, description, formula
        )

    @staticmethod
    def create_subplot_grid(data: Dict[str, Any],
                          case: BaseCase,
                          case_name: str,
                          title: str | None = None,
                          y_label: str = None,
                          description: str = None,
                          cols: int = 2,
                          config: ChartConfig = None) -> str:
        """Create a subplot grid with automatic styling"""

        if data is None:
            return '<div class="alert alert-warning">Nothing to plot (padding >= total_epochs).</div>'

        builder = PlotlyChartBuilder(config)
        config = builder.config

        # Setup subplot grid
        num_charts = len(data.get('subset_titles', []))
        rows = math.ceil(num_charts / cols)
        vertical_spacing = 25 / config.height
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=data.get('subset_titles', []),
            vertical_spacing=vertical_spacing,
            horizontal_spacing=0.1,
            specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
        )
        fig.update_annotations(font_size=10)
        # Get emphasized validator
        emphasized_validator = getattr(case, 'shift_validator_hotkey', None)

        # Build all trace dicts first, then add in bulk using dictionaries only
        trace_dicts = []
        row_indices = []
        col_indices = []

        for i_chart in range(num_charts):
            row = (i_chart // cols) + 1
            col = (i_chart % cols) + 1

            chart_data = data['chart_data'][i_chart]

            for j, series in enumerate(chart_data['series']):
                style = builder._create_series_style(
                    series['key'],
                    j,
                    case,
                    emphasized_validator
                )

                # Only show legend for first subplot
                style.show_in_legend = (i_chart == 0)

                if 'name' in series:
                    name = series['name']
                elif hasattr(case, 'hotkey_label_map'):
                    name = case.hotkey_label_map.get(series['key'], series['key'][:10] + '...')
                else:
                    name = series['key']

                trace_dict = builder._create_trace_dict(
                    data['x'],
                    series['y_data'],
                    name,
                    style
                )

                trace_dicts.append(trace_dict)
                row_indices.append(row)
                col_indices.append(col)

        # Add all traces at once using dictionaries directly
        fig.add_traces(trace_dicts, rows=row_indices, cols=col_indices)

        # Update layout
        height = rows * config.height
        layout = builder._create_base_layout(height=height)

        # Update subplot axes in bulk
        axis_updates = {}
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                subplot_num = (i-1)*cols + j
                if subplot_num > num_charts:
                    continue

                xaxis_key = f'xaxis{subplot_num}' if subplot_num > 1 else 'xaxis'
                yaxis_key = f'yaxis{subplot_num}' if subplot_num > 1 else 'yaxis'

                axis_updates[xaxis_key] = {
                    **layout['xaxis'],
                    'title': "Epoch"
                }
                axis_updates[yaxis_key] = {
                    **layout['yaxis'],
                    'title': y_label if j == 1 else "",
                    'range': data.get('y_range', [None, None])
                }

        layout.update(axis_updates)
        fig.update_layout(layout)

        return builder._generate_html(
            fig, title, case_name, description
        )


# Simplified plotting functions using the factory
def plot_relative_dividends_plotly(
    validators_relative_dividends: dict[str, list[float]],
    case_name: str,
    case: BaseCase,
    num_epochs: int,
    epochs_padding: int = 0,
    **kwargs,
) -> str:

    data = _prepare_relative_dividends_data(
        validators_relative_dividends, case, num_epochs, epochs_padding
    )

    description, _, formula_text = _get_relative_dividends_description_and_formula()

    # Transform data for percentage display
    if data:
        for series_info in data['series_data']:
            series_info['data'] = series_info['data'] * 100

    return ChartFactory.create_line_chart(
        data=data,
        case=case,
        case_name=case_name,
        title="Validators Relative Dividends",
        y_label="Relative Dividend (%)",
        description=description,
        formula=f"<strong>Relative Dividend =</strong> <code>{formula_text}</code>",
        config=ChartConfig(grid_legend=True),
    )


def plot_bonds_metagraph_dynamic_plotly(
    case: MetagraphCase,
    bonds_per_epoch: list[torch.Tensor],
    default_miners: list[str],
    case_name: str,
    normalize: bool = False,
    legend_validators: list[str] | None = None,
    epochs_padding: int = 0,
    **kwargs,
) -> str:

    raw_data = _prepare_bonds_metagraph_data(
        case, bonds_per_epoch, default_miners, normalize, epochs_padding
    )

    description, ylabel, title_suffix = _get_bonds_description_and_labels(normalize)

    # Adapt data using common function with ugly key parametrization
    if raw_data:
        raw_data['y_range'] = [0, 1.05] if normalize else [0, None]
        subplot_data = _adapt_data_for_subplots(
            raw_data,
            chart_titles_key='subset_m',
            series_keys_key='subset_v',
            data_matrix_key='plot_data',
        )
    else:
        subplot_data = None

    return ChartFactory.create_subplot_grid(
        data=subplot_data,
        case=case,
        case_name=case_name,
        title=f"Validators Bonds per Miner{title_suffix}",
        y_label=ylabel,
        description=description,
        config=ChartConfig(height=250),
    )


def plot_validator_server_weights_subplots_dynamic_plotly(
    case: MetagraphCase,
    default_miners: list[str],
    case_name: str,
    epochs_padding: int = 0,
    **kwargs,
) -> str:

    raw_data = _prepare_validator_server_weights_subplots_dynamic_data(case, default_miners, epochs_padding)
    description = _get_validator_weights_description()

    # Adapt data using common function with ugly key parametrization
    if raw_data:
        raw_data['y_range'] = [0, 1.05]
        subplot_data = _adapt_data_for_subplots(
            raw_data,
            chart_titles_key='subset_srvs',
            series_keys_key='subset_vals',
            data_matrix_key='data_cube',
        )
    else:
        subplot_data = None

    return ChartFactory.create_subplot_grid(
        data=subplot_data,
        case=case,
        case_name=case_name,
        title="Validators Weights per Miner",
        y_label="Validator Weight",
        description=description,
        config=ChartConfig(height=250),
    )


def plot_validator_server_weights_subplots_plotly(
    validators: list[str],
    weights_epochs: list[torch.Tensor],
    servers: list[str],
    num_epochs: int,
    case_name: str,
    epochs_padding: int = 0,
    **kwargs,
) -> str:
    """
    Plots validator weights in subplots (one subplot per server) over epochs using plotly.
    Each subplot shows lines for all validators, representing how much weight
    they allocate to that server from epoch 0..num_epochs-1.
    """
    raw_data = _prepare_validator_server_weights_subplots_data(
        validators, weights_epochs, servers, num_epochs, epochs_padding
    )

    # Adapt data for subplot grid factory
    if raw_data:
        subplot_data = _adapt_data_for_subplots(
            raw_data,
            chart_titles_key='servers',
            series_keys_key='validators',
            data_matrix_key='data_matrix',
        )
    else:
        subplot_data = None

    return ChartFactory.create_subplot_grid(
        data=subplot_data,
        case=None,  # No case object available in this context
        case_name=case_name,
        title="Validators Weights per Server",
        y_label="Validator Weight",
        cols=min(len(servers), 3),
        config=ChartConfig(height=250),
    )


def _dividends_special_case_handler(layout: dict, case_name: str, data: dict) -> dict:
    """Special case handler for dividends charts"""
    # Special case handling for Case 4
    if case_name.startswith("Case 4"):
        layout['yaxis']['range'] = [0, 0.042]
    return layout


# Plotly implementation using ChartFactory
def plot_dividends_plotly(
    num_epochs: int,
    validators: list[str],
    dividends_per_validator: dict[str, list[float]],
    case_name: str,
    case: BaseCase,
    **kwargs,
) -> str:
    """
    Generates a plotly plot of dividends over epochs for a set of validators.
    """

    data = _prepare_dividends_data(num_epochs, validators, dividends_per_validator, case)

    if data is None:
        return '<div class="alert alert-warning">No dividend data to plot.</div>'

    # Create custom config for scientific notation and x-shifting
    config = ChartConfig(
        height=500,
        legend_horizontal=True,
        grid_legend=True,
        y_axis_format="auto",
        y_axis_range=(0, None),  # Start from zero
        enable_x_shift=True,
        x_shift_delta=0.05,
    )

    return ChartFactory.create_line_chart(
        data=data,
        case=case,
        case_name=case_name,
        y_label="Dividend per 1,000 TAO per Epoch",
        config=config,
        special_case_handler=_dividends_special_case_handler
    )


def plot_bonds_plotly(
    num_epochs: int,
    validators: list[str],
    servers: list[str],
    bonds_per_epoch: list[torch.Tensor],
    case_name: str,
    normalize: bool = False,
    **kwargs,
) -> str:
    """Generates a plotly plot of bonds per server for each validator."""

    # Process data
    bonds_data = _prepare_bonds_data(bonds_per_epoch, validators, servers, normalize=normalize)
    description, ylabel, title_suffix = _get_bonds_description_and_labels(normalize)

    # Transform data for ChartFactory
    chart_data = []
    for s_idx, server in enumerate(servers):
        series_list = []
        for v_idx, validator in enumerate(validators):
            series_list.append({
                'key': validator,
                'y_data': bonds_data[s_idx][v_idx],
                'name': validator
            })
        chart_data.append({'series': series_list})

    subplot_data = {
        'subset_titles': servers,
        'chart_data': chart_data,
        'x': list(range(num_epochs)),
        'y_range': [0, 1.05] if normalize else (0, None)
    }

    # Create config with appropriate settings
    config = ChartConfig(
        height=250,
        legend_horizontal=True,
        y_axis_range=(0, 1.05) if normalize else (0, None)
    )

    return ChartFactory.create_subplot_grid(
        data=subplot_data,
        case=None,  # No case object available
        case_name=case_name,
        title=f"Validators Bonds per Server{title_suffix}",
        y_label=ylabel,
        description=None,  # no description
        cols=min(len(servers), 3),
        config=config
    )


def _validator_server_weights_special_handler_plotly(layout: dict, case_name: str, data: dict) -> dict:
    """Special plotly handler for validator server weights."""
    # Apply custom y-axis ticks
    layout['yaxis'].update({
        'tickmode': 'array',
        'tickvals': data['y_tick_positions'],
        'ticktext': data['y_tick_labels'],
        'range': data['y_range']
    })

    return layout


def plot_validator_server_weights_plotly(
    validators: list[str],
    weights_epochs: list[torch.Tensor],
    servers: list[str],
    num_epochs: int,
    case_name: str,
    **kwargs,
) -> str:
    """Plotly implementation of validator server weights plot."""

    data = _prepare_validator_server_weights_data(
        validators, weights_epochs, servers, num_epochs
    )

    if data is None:
        return '<div class="alert alert-warning">No weights data to plot.</div>'

    # Create config with adaptive height
    base_height = 200 if data['fig_height_scale'] == 1 else 300
    config = ChartConfig(
        height=base_height,
        legend_horizontal=True,
        y_axis_range=tuple(data['y_range']),
    )

    return ChartFactory.create_line_chart(
        data=data,
        case=None,  # No case object available
        case_name=case_name,
        title="Validators Weights to Servers",
        config=config,
        special_case_handler=_validator_server_weights_special_handler_plotly
    )


def plot_incentives_plotly(
    servers: list[str],
    server_incentives_per_epoch: list[torch.Tensor],
    num_epochs: int,
    case_name: str,
    case: BaseCase = None,
    **kwargs,
) -> str:
    """Generates a plotly plot of server incentives over epochs."""

    data = _prepare_incentives_data(servers, server_incentives_per_epoch, num_epochs, case)

    if data is None:
        return '<div class="alert alert-warning">No incentive data to plot.</div>'

    # Create config with fixed y-axis range for incentives
    config = ChartConfig(
        height=300,
        legend_horizontal=True,
        y_axis_range=(0, 1.05),
        grid_legend=False,
    )

    return ChartFactory.create_line_chart(
        data=data,
        case=case,
        case_name=case_name,
        title="Server Incentives Over Time",
        y_label="Server Incentive",
        config=config,
        series_key='server',
    )
