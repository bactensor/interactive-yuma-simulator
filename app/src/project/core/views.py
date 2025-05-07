import json
import re
from dataclasses import asdict

import pandas as pd
from django.http import HttpResponse
from django.shortcuts import render
from yuma_simulation._internal.cases import get_synthetic_cases, instantiate_metagraph_case
from yuma_simulation._internal.yumas import SimulationHyperparameters, YumaParams, YumaSimulationNames
from yuma_simulation.v1 import api as yuma_api
from yuma_simulation.v1.api import generate_chart_table, generate_metagraph_based_chart_table

from .forms import SelectionForm, SimulationHyperparametersForm, YumaParamsForm
from .utils import ONE_MILLION, UINT16_MAX, normalize, fetch_metagraph_data

import logging
logger = logging.getLogger(__name__)

def simulation_view(request):
    logger.info("ENTER simulation_view with GET params: %s", request.GET.dict())

    selection_form = SelectionForm(request.GET or None)
    hyper_form     = SimulationHyperparametersForm(request.GET or None)
    yuma_form      = YumaParamsForm(request.GET or None)

    logger.info(
        "  raw forms valid? sel=%s, hyp=%s, yuma=%s",
        selection_form.is_valid(),
        hyper_form.is_valid(),
        yuma_form.is_valid(),
    )

    context = {
        "selection_form": selection_form,
        "hyper_form":     hyper_form,
        "yuma_form":      yuma_form,
        "valid_forms":    False,
        "cases_json":     "[]",
        "yumas_json":     "{}",
    }

    use_mg = request.GET.get("use_metagraph") == "on"
    sel_ok = selection_form.is_valid() or use_mg
    hyp_ok = hyper_form.is_valid()
    yum_ok = yuma_form.is_valid()

    if sel_ok and hyp_ok and yum_ok:
        context["valid_forms"] = True

        if selection_form.is_valid():
            cases = selection_form.cleaned_data["selected_cases"]
        else:
            cases = []
        context["cases_json"] = json.dumps(cases)

        yumas_dict      = asdict(YumaSimulationNames())
        selected_key    = selection_form.cleaned_data.get("selected_yumas")
        yuma_data       = yuma_form.cleaned_data
        if selected_key == "YUMA3" and yuma_data.get("liquid_alpha"):
            selected_key += "_LIQUID"
        chosen_yuma     = yumas_dict[selected_key]

        hyper          = hyper_form.cleaned_data
        sim_params     = {
            "kappa":                        hyper["kappa"],
            "bond_penalty":                 hyper["bond_penalty"],
            "reset_bonds":                  hyper["reset_bonds"],
            "liquid_alpha_consensus_mode":  hyper["liquid_alpha_consensus_mode"],
        }

        yp             = yuma_data
        yuma_params    = {
            "bond_moving_avg":         yp["bond_moving_avg"],
            "liquid_alpha":            yp["liquid_alpha"],
            "alpha_high":              yp["alpha_high"],
            "alpha_low":               yp["alpha_low"],
            "decay_rate":              yp["decay_rate"],
            "capacity_alpha":          yp["capacity_alpha"],
            "alpha_sigmoid_steepness": yp["alpha_sigmoid_steepness"],
        }

        context["yumas_json"] = json.dumps({
            "selected_yuma_key": selected_key,
            "chosen_yuma":       chosen_yuma,
            "sim_params":        sim_params,
            "yuma_params":       yuma_params,
        })

    return render(request, "simulator.html", context)


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

        chosen_yuma = request.GET.get("chosen_yuma", "YUMA")

        raw_bond_moving_avg = float(request.GET.get("bond_moving_avg", 900_000))
        alpha_high = float(request.GET.get("alpha_high", 0.3))
        alpha_low = float(request.GET.get("alpha_low", 0.1))
        decay_rate = float(request.GET.get("decay_rate", 0.1))
        capacity_alpha = float(request.GET.get("capacity_alpha", 0.1))
        alpha_sigmoid_steepness = float(request.GET.get("alpha_sigmoid_steepness", 10.0))

    except ValueError as e:
        return HttpResponse(f"Invalid parameter: {str(e)}", status=400)

    sim_params = SimulationHyperparameters(
        kappa=normalize(raw_kappa, UINT16_MAX),
        bond_penalty=normalize(raw_bond_penalty, UINT16_MAX),
        liquid_alpha_consensus_mode=liquid_alpha_consensus_mode,
    )

    yuma_params = YumaParams(
        bond_moving_avg=normalize(raw_bond_moving_avg, ONE_MILLION),
        liquid_alpha=liquid_alpha,
        alpha_high=alpha_high,
        alpha_low=alpha_low,
        decay_rate=decay_rate,
        capacity_alpha=capacity_alpha,
        alpha_sigmoid_steepness=alpha_sigmoid_steepness,
    )

    package_cases = get_synthetic_cases(use_full_matrices=True, reset_bonds=reset_bonds)
    chosen_case = next((c for c in package_cases if c.name == case_name), None)
    if not chosen_case:
        return HttpResponse(f"Case '{case_name}' not found.", status=404)

    try:
        selected_yumas = [(chosen_yuma, yuma_params)]
        partial_html = generate_chart_table(
            cases=[chosen_case], yuma_versions=selected_yumas, yuma_hyperparameters=sim_params
        )
    except Exception as e:
        return HttpResponse(f"Error generating chart: {str(e)}", status=500)

    return HttpResponse(partial_html.data if partial_html else "No data", status=200)

def metagraph_simulation_view(request):
    logger.info("ENTER metagraph_simulation_view GET: %s", request.GET.dict())

    try:
        raw_kappa = float(request.GET.get("kappa", 32767))
        raw_bond_penalty = float(request.GET.get("bond_penalty", 65535))
        reset_bonds = request.GET.get("reset_bonds", "False") == "true"
        liquid_alpha_consensus_mode = request.GET.get("liquid_alpha_consensus_mode", "CURRENT")
        liquid_alpha = request.GET.get("liquid_alpha", "False") == "true"

        chosen_yuma = request.GET.get("selected_yumas", "YUMA")

        raw_bond_moving_avg = float(request.GET.get("bond_moving_avg", 900_000))
        alpha_high = float(request.GET.get("alpha_high", 0.3))
        alpha_low = float(request.GET.get("alpha_low", 0.1))
        decay_rate = float(request.GET.get("decay_rate", 0.1))
        capacity_alpha = float(request.GET.get("capacity_alpha", 0.1))
        alpha_sigmoid_steepness = float(request.GET.get("alpha_sigmoid_steepness", 10.0))

        start_block = int(request.GET.get("start_block", 0))
        epochs_num = int(request.GET.get("epochs_num",   0))
        netuid = int(request.GET.get("netuid",      0))
        validators = request.GET.get("validators", "").split(",")

    except ValueError as e:
        return HttpResponse(f"Invalid parameter: {str(e)}", status=400)

    sim_params = SimulationHyperparameters(
        kappa=normalize(raw_kappa, UINT16_MAX),
        bond_penalty=normalize(raw_bond_penalty, UINT16_MAX),
        liquid_alpha_consensus_mode=liquid_alpha_consensus_mode,
    )

    yuma_params = YumaParams(
        bond_moving_avg=normalize(raw_bond_moving_avg, ONE_MILLION),
        liquid_alpha=liquid_alpha,
        alpha_high=alpha_high,
        alpha_low=alpha_low,
        decay_rate=decay_rate,
        capacity_alpha=capacity_alpha,
        alpha_sigmoid_steepness=alpha_sigmoid_steepness,
    )

    try:
        logger.info("→ fetch_metagraph_data(start=%s, end=%s, netuid=%s)",
                    start_block, start_block + epochs_num * 360, netuid)
        mg_data = fetch_metagraph_data(
            start_block=start_block,
            end_block=start_block + (epochs_num * 360),
            netuid=netuid,
        )
        logger.info("← fetch_metagraph_data returned %s records", len(mg_data) if isinstance(mg_data, dict) else '??')
    except Exception as e:
        logger.error("fetch_metagraph_data FAILED", exc_info=True)
        return HttpResponse(f"Error fetching metagraph metadata: {e}", status=400)

    top_validators_ids = []
    for vali_idx in validators:
        top_validators_ids.append((int(vali_idx)))

    case = instantiate_metagraph_case(
        mg_data=mg_data,
        top_validators_ids=top_validators_ids,
    )

    yumas_dict = asdict(YumaSimulationNames())
    chosen_yuma = yumas_dict[chosen_yuma]

    selected_yumas = [(chosen_yuma, yuma_params)]

    logger.info("→ calling generate_metagraph_based_chart_table()")
    partial_html = generate_metagraph_based_chart_table(
        yuma_versions=selected_yumas,
        normal_case=case,
        yuma_hyperparameters=sim_params,
        epochs_padding=0,
    )

    logger.info("← generate_metagraph_based_chart_table done; html length=%s", len(partial_html.data))
    return HttpResponse(partial_html.data if partial_html else "No data", status=200)

def bootstrap_generate_ipynb_table(
    table_data: dict[str, list[str]],
    summary_table: pd.DataFrame | None,
    case_row_ranges: list[tuple[int, int, int]],
) -> str:
    if summary_table is None:
        summary_table = pd.DataFrame(table_data)

    # helper: extract just the src URL from the <img> tag
    def parse_img_src(html_str: str) -> str:
        m = re.search(r'src="([^"]+)"', html_str)
        return m.group(1) if m else ""

    rows = []
    num_rows = len(summary_table)

    for i in range(num_rows):
        # figure out which case this row belongs to
        case_name = next(
            (summary_table.columns[c_idx] for start, end, c_idx in case_row_ranges if start <= i <= end),
            summary_table.columns[0],
        )

        raw_img_tag = summary_table.at[i, case_name]
        img_src = parse_img_src(raw_img_tag)

        # full-width, single-row layout:
        rows.append(f"""
          <div class="mb-4 text-center">
            <img src="{img_src}"
                 class="img-fluid w-100"
                 alt="Chart for {case_name}">
          </div>
        """)

    # wrap everything in a single container
    return f"""
    <div class="container-fluid px-3">
      {"".join(rows)}
    </div>
    """


# Now override the external lib’s function:
yuma_api._generate_ipynb_table = bootstrap_generate_ipynb_table
