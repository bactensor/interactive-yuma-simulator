import json
import logging
import re
from dataclasses import asdict
from datetime import datetime, timedelta
from functools import lru_cache
from django.http import HttpResponseBadRequest
from django.conf import settings


import pandas as pd
from django.http import HttpResponse, HttpResponseServerError, JsonResponse
from django.shortcuts import render
from django.views.decorators.cache import cache_control
from requests.exceptions import HTTPError, Timeout
from project.yuma_simulation._internal.cases import get_synthetic_cases, MetagraphCase
from project.yuma_simulation._internal.yumas import SimulationHyperparameters, YumaParams, YumaSimulationNames
from project.yuma_simulation.v1 import api as yuma_api
from project.yuma_simulation.v1.api import generate_chart_table, generate_metagraph_based_chart_table

from .forms import SelectionForm, SimulationHyperparametersForm, YumaParamsForm
from .utils import ONE_MILLION, UINT16_MAX, fetch_metagraph_data, normalize

logger = logging.getLogger(__name__)


def simulation_view(request):
    selection_form = SelectionForm(request.GET or None)
    hyper_form = SimulationHyperparametersForm(request.GET or None)
    yuma_form = YumaParamsForm(request.GET or None, selected_yuma=request.GET.get('selected_yumas'),)

    context = {
        "selection_form": selection_form,
        "hyper_form": hyper_form,
        "yuma_form": yuma_form,
        "valid_forms": False,
        "cases_json": "[]",
        "yumas_json": "{}",
    }

    if request.GET.get("use_metagraph"):
        selection_form.fields["selected_cases"].required = False

    if selection_form.is_valid() and hyper_form.is_valid() and yuma_form.is_valid():
        cleaned = selection_form.cleaned_data

        if selection_form.is_valid():
            context["valid_forms"] = True

            selected_case_names = cleaned.get("selected_cases", [])
            context["cases_json"] = json.dumps(selected_case_names)

            yumas_dict = asdict(YumaSimulationNames())
            yuma_key = cleaned["selected_yumas"]
            yuma_data = yuma_form.cleaned_data

            if yuma_key == "YUMA3" and yuma_data["liquid_alpha"]:
                yuma_key += "_LIQUID"

            chosen_yuma = yumas_dict[yuma_key]
            hyper_data = hyper_form.cleaned_data

            sim_params = {
                "kappa": hyper_data["kappa"],
                "bond_penalty": hyper_data["bond_penalty"],
                "reset_bonds": hyper_data["reset_bonds"],
                "liquid_alpha_consensus_mode": hyper_data["liquid_alpha_consensus_mode"],
                "alpha_tao_ratio": hyper_data["alpha_tao_ratio"],
            }
            yuma_params = {
                "bond_moving_avg": yuma_data["bond_moving_avg"],
                "liquid_alpha": yuma_data["liquid_alpha"],
                "alpha_high": yuma_data["alpha_high"],
                "alpha_low": yuma_data["alpha_low"],
                "decay_rate": yuma_data["decay_rate"],
                "capacity_alpha": yuma_data["capacity_alpha"],
                "alpha_sigmoid_steepness": yuma_data["alpha_sigmoid_steepness"],
            }

            effective_weights = selection_form.cleaned_data.get("liquid_alpha_effective_weights", False)
            if yuma_key.startswith("YUMA3"):
                yuma_params["liquid_alpha_effective_weights"] = effective_weights

            context["yumas_json"] = json.dumps(
                {
                    "selected_yuma_key": yuma_key,
                    "chosen_yuma": chosen_yuma,
                    "shifted_validator_hotkey": cleaned["shifted_validator_hotkey"],
                    "sim_params": sim_params,
                    "yuma_params": yuma_params,
                }
            )
            context["cache_key"] = settings.CACHE_KEY
            context["jsCharts"] = request.GET.get('jsCharts', 'off')

    return render(request, "simulator.html", context)


@cache_control(public=True, max_age=604800, s_maxage=604800)
def simulate_single_case_view(request):
    """
    Returns HTML snippet for a single case simulation.
    Expects query parameters for case_name, sim_params, yuma_params, etc.
    """
    case_name = request.GET.get("case_name")
    if not case_name:
        return HttpResponse("Missing 'case_name' parameter.", status=400)

    try:
        raw_kappa = float(request.GET.get("kappa", 32767))
        raw_bond_penalty = float(request.GET.get("bond_penalty", 65535))
        reset_bonds = request.GET.get("reset_bonds", "False") == "true"
        liquid_alpha_consensus_mode = request.GET.get("liquid_alpha_consensus_mode", "CURRENT")
        liquid_alpha = request.GET.get("liquid_alpha", "False") == "true"
        effective_weights = request.GET.get("liquid_alpha_effective_weights", "False") == "true"

        chosen_yuma = request.GET.get("chosen_yuma", "YUMA")

        raw_bond_moving_avg = float(request.GET.get("bond_moving_avg", 900_000))
        alpha_high = float(request.GET.get("alpha_high", 0.3))
        alpha_low = float(request.GET.get("alpha_low", 0.1))
        decay_rate = float(request.GET.get("decay_rate", 0.1))
        capacity_alpha = float(request.GET.get("capacity_alpha", 0.1))
        alpha_sigmoid_steepness = float(request.GET.get("alpha_sigmoid_steepness", 10.0))
        js_charts = request.GET.get('jsCharts', '') == 'on'

    except ValueError as e:
        return HttpResponse(f"Invalid parameter: {str(e)}", status=400)

    sim_params = SimulationHyperparameters(
        kappa=normalize(raw_kappa, UINT16_MAX),
        bond_penalty=normalize(raw_bond_penalty, UINT16_MAX),
        liquid_alpha_consensus_mode=liquid_alpha_consensus_mode,
    )

    yuma_kwargs = {
        "bond_moving_avg":             normalize(raw_bond_moving_avg, ONE_MILLION),
        "liquid_alpha":                liquid_alpha,
        "alpha_high":                  alpha_high,
        "alpha_low":                   alpha_low,
        "decay_rate":                  decay_rate,
        "capacity_alpha":              capacity_alpha,
        "alpha_sigmoid_steepness":     alpha_sigmoid_steepness,
    }
    if chosen_yuma.startswith("YUMA3"):
        yuma_kwargs["liquid_alpha_effective_weights"] = effective_weights

    yuma_params = YumaParams(**yuma_kwargs)

    package_cases = get_synthetic_cases(use_full_matrices=True, reset_bonds=reset_bonds)
    chosen_case = next((c for c in package_cases if c.name == case_name), None)
    if not chosen_case:
        return HttpResponse(f"Case '{case_name}' not found.", status=404)

    try:
        selected_yumas = [(chosen_yuma, yuma_params)]
        partial_html = generate_chart_table(
            cases=[chosen_case],
            yuma_versions=selected_yumas,
            yuma_hyperparameters=sim_params,
            engine='plotly' if js_charts else 'matplotlib',
        )
    except Exception as e:
        return HttpResponse(f"Error generating chart: {str(e)}", status=500)

    return HttpResponse(partial_html.data if partial_html else "No data", status=200)


@cache_control(public=True, max_age=604800, s_maxage=604800)
def metagraph_simulation_view(request):
    try:
        raw_kappa = float(request.GET.get("kappa", 32767))
        raw_bond_penalty = float(request.GET.get("bond_penalty", 65535))
        lam = request.GET.get("liquid_alpha_consensus_mode", "CURRENT")
        liq_alpha = request.GET.get("liquid_alpha", "False") == "true"
        chosen_yuma = request.GET.get("selected_yumas", "YUMA")
        effective_weights = request.GET.get("liquid_alpha_effective_weights", "False") == "true"


        raw_bma = float(request.GET.get("bond_moving_avg", 900_000))
        alpha_high = float(request.GET.get("alpha_high", 0.3))
        alpha_low = float(request.GET.get("alpha_low", 0.1))
        decay_rate = float(request.GET.get("decay_rate", 0.1))
        cap_alpha = float(request.GET.get("capacity_alpha", 0.1))
        steepness = float(request.GET.get("alpha_sigmoid_steepness", 10.0))

        raw_start = request.GET.get("start_date")
        raw_end   = request.GET.get("end_date")
        js_charts = request.GET.get('jsCharts', '') == 'on'
        shifted_validator_hotkey = request.GET.get('shifted_validator_hotkey', '')

        try:
            start_date = datetime.fromisoformat(raw_start) if raw_start else None
            end_date   = datetime.fromisoformat(raw_end)   if raw_end   else None
        except ValueError:
            return HttpResponseBadRequest("Dates must be in YYYY-MM-DD format.")
        netuid = int(request.GET.get("netuid", 0))
        requested_miners = [m.strip()
                    for m in request.GET.getlist("miners_hotkeys")
                    if m.strip()]

        raw_alpha_tao = float(request.GET.get("alpha_tao_ratio", 0.1))
    except ValueError as e:
        return HttpResponse(f"Invalid parameter: {e}", status=400)

    sim_params = SimulationHyperparameters(
        kappa=normalize(raw_kappa, UINT16_MAX),
        bond_penalty=normalize(raw_bond_penalty, UINT16_MAX),
        liquid_alpha_consensus_mode=lam,
        alpha_tao_ratio=raw_alpha_tao,
    )

    mg_yuma_kwargs = {
        "bond_moving_avg":             normalize(raw_bma, ONE_MILLION),
        "liquid_alpha":                liq_alpha,
        "alpha_high":                  alpha_high,
        "alpha_low":                   alpha_low,
        "decay_rate":                  decay_rate,
        "capacity_alpha":              cap_alpha,
        "alpha_sigmoid_steepness":     steepness,
    }

    if chosen_yuma.startswith("YUMA3"):
        mg_yuma_kwargs["liquid_alpha_effective_weights"] = effective_weights

    yuma_params = YumaParams(**mg_yuma_kwargs)

    epochs_padding = int(settings.EPOCHS_PADDING)
    start_date = start_date - timedelta(seconds=360 * 12 * epochs_padding)
    try:
        mg_data = fetch_metagraph_data(
            start_date=start_date,
            end_date=end_date,
            netuid=netuid,
        )
    except Timeout:
        return HttpResponseServerError(
            "The service timed out."
            "Try requesting fewer epochs."
        )
    except HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return HttpResponse(f"No metagraph data for timespan {start_date}–{end_date}", status=404)
        if e.response is not None and e.response.status_code >= 500:
            html = """
            <div class="alert alert-danger">
              <strong>Internal Server Error</strong>
              <ul class="mb-0">
                <li>Make sure you’re querying historical metagraph data no older than 35 days ago.</li>
              </ul>
            </div>
            """
            return HttpResponse(html, status=500)
        return HttpResponse(f"Error fetching metagraph data: {e}", status=400)
    except Exception as e:
        return HttpResponse(f"Error fetching metagraph data: {e}", status=400)

    case_config = {}
    if shifted_validator_hotkey:
        case_config.update({
            'introduce_shift': True,
            'shift_validator_hotkey': shifted_validator_hotkey,
        })
    try:
        case, invalid_miners = MetagraphCase.from_mg_dumper_data(
            mg_data=mg_data,
            requested_miners=requested_miners,
            **case_config,
            )
    except ValueError as e:
        return HttpResponse(str(e), status=400)

    selection_form = SelectionForm(request.GET or None)
    for hotkey in invalid_miners:
        selection_form.add_error(
            "miners_hotkeys",
            f"Invalid miner hotkey: {hotkey}"
        )

    selected_chart_yumas = [(asdict(YumaSimulationNames())[chosen_yuma], yuma_params)]
    all_yumas = list(asdict(YumaSimulationNames()).values())
    summary_versions = [
        (display_name, yuma_params)
        for display_name in all_yumas
        if display_name not in ("Yuma 1", "Yuma 3 (Rhef+relative bonds)")
    ]

    partial_html = generate_metagraph_based_chart_table(
        chart_versions=selected_chart_yumas,
        summary_versions=summary_versions,
        normal_case=case,
        yuma_hyperparameters=sim_params,
        epochs_padding=epochs_padding,
        engine='plotly' if js_charts else 'matplotlib',
    )

    return JsonResponse(
        {
            "html": partial_html.data or "No data",
            "errors": selection_form.errors,
        },
        status=200
    )

def bootstrap_generate_ipynb_table(
    table_data: dict[str, list[str]],
    summary_table: pd.DataFrame | None,
    case_row_ranges: list[tuple[int, int, int]],
) -> str:
    if summary_table is None:
        summary_table = pd.DataFrame(table_data)

    cols = list(summary_table.columns)
    rows_html: list[str] = []

    def parse_img_src(html_str: str) -> str:
        m = re.search(r'src="([^"]+)"', html_str)
        return m.group(1) if m else ""

    for i in range(len(summary_table)):
        # find a matching (start,end,c_idx) where c_idx is in-bounds
        match = next(
            ((start, end, c_idx) for start, end, c_idx in case_row_ranges
             if start <= i <= end and 0 <= c_idx < len(cols)),
            None
        )
        if match:
            _, _, c_idx = match
        else:
            # fallback to the first column
            c_idx = 0

        case_name = cols[c_idx]
        raw_html = summary_table.iat[i, c_idx]
        img_src = parse_img_src(raw_html)

        if img_src:
            rows_html.append(f'''
              <div class="mb-4 text-center">
                <img src="{img_src}"
                     class="img-fluid w-100"
                     alt="Chart for {case_name}">
              </div>
            ''')
        else:
            rows_html.append(f'''
              <div class="mb-4 text-center">
                {raw_html}
              </div>
            ''')

    return f'''
    <div class="container-fluid px-3">
      {''.join(rows_html)}
    </div>
    '''


yuma_api._generate_ipynb_table = bootstrap_generate_ipynb_table