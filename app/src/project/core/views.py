# views.py

import json
import traceback

from django.shortcuts import render
from django.http import HttpResponse
# from django.views.decorators.cache import cache_page
from dataclasses import asdict

from .forms import SelectionForm, SimulationHyperparametersForm, YumaParamsForm

from yuma_simulation._internal.cases import get_synthetic_cases
from yuma_simulation._internal.yumas import (
    SimulationHyperparameters,
    YumaParams,
    YumaSimulationNames
)
from yuma_simulation.v1.api import generate_chart_table

def simulation_view(request):
    # 1) Instantiate forms with request.GET (or request.POST), or empty.
    selection_form = SelectionForm(request.GET or None)
    hyper_form = SimulationHyperparametersForm(request.GET or None)
    yuma_form = YumaParamsForm(request.GET or None)

    # We'll pass these to the template
    context = {
        "selection_form": selection_form,
        "hyper_form": hyper_form,
        "yuma_form": yuma_form,
        "valid_forms": False,
        "cases_json": "[]",  # default empty JSON list
        "yumas_json": "{}",  # default empty JSON object
    }

    if selection_form.is_valid() and hyper_form.is_valid() and yuma_form.is_valid():
        context["valid_forms"] = True

        # Extract selected cases
        selected_case_names = selection_form.cleaned_data["selected_cases"]
        # Convert to JSON
        context["cases_json"] = json.dumps(selected_case_names)

        # Determine selected YUMA
        yumas_dict = asdict(YumaSimulationNames())
        selected_yuma_key = selection_form.cleaned_data["selected_yumas"]
        
        yuma_data = yuma_form.cleaned_data
        if selected_yuma_key in ["YUMA", "YUMA4"] and yuma_data["liquid_alpha"]:
            selected_yuma_key += "_LIQUID"

        chosen_yuma = yumas_dict[selected_yuma_key]

        # Prepare sim_params
        hyper_data = hyper_form.cleaned_data
        sim_params = {
            "kappa": hyper_data["kappa"],
            "bond_penalty": hyper_data["bond_penalty"],
            "reset_bonds": hyper_data["reset_bonds"],
            "liquid_alpha_consensus_mode": hyper_data["liquid_alpha_consensus_mode"],
        }

        # Prepare yuma_params
        yuma_params = {
            "bond_moving_avg": yuma_data["bond_moving_avg"],
            "liquid_alpha": yuma_data["liquid_alpha"],
            "alpha_high": yuma_data["alpha_high"],
            "alpha_low": yuma_data["alpha_low"],
            "decay_rate": yuma_data["decay_rate"],
            "capacity_alpha": yuma_data["capacity_alpha"],
            "alpha_sigmoid_steepness": yuma_data["alpha_sigmoid_steepness"],
            "override_consensus_high": yuma_data["override_consensus_high"],
            "override_consensus_low": yuma_data["override_consensus_low"],
        }

        # Build a dict for the front end
        yumas_json_data = {
            "selected_yuma_key": selected_yuma_key,
            "chosen_yuma": chosen_yuma,
            "sim_params": sim_params,
            "yuma_params": yuma_params,
        }

        context["yumas_json"] = json.dumps(yumas_json_data)

    return render(request, "simulator.html", context)

def simulate_single_case_view(request):
    """
    Returns HTML snippet for a single case simulation.
    Expects query parameters for case_name, sim_params, yuma_params, etc.
    """
    case_name = request.GET.get("case_name")
    if not case_name:
        return HttpResponse("Missing 'case_name' parameter.", status=400)

    # 1) Parse simulation hyperparams from GET
    try:
        kappa = float(request.GET.get("kappa", 0.5))
        bond_penalty = float(request.GET.get("bond_penalty", 1.0))
        reset_bonds = request.GET.get("reset_bonds", "False") == "True"
        liquid_alpha_consensus_mode = request.GET.get("liquid_alpha_consensus_mode", "CURRENT")

        # 2) Parse YUMA version and its parameters
        selected_yuma_key = request.GET.get("selected_yuma_key", "YUMA")
        chosen_yuma = request.GET.get("chosen_yuma", "YUMA")

        bond_moving_avg = float(request.GET.get("bond_moving_avg", 0.975))
        liquid_alpha = request.GET.get("liquid_alpha", "False") == "True"
        alpha_high = float(request.GET.get("alpha_high", 0.3))
        alpha_low = float(request.GET.get("alpha_low", 0.1))
        decay_rate = float(request.GET.get("decay_rate", 0.1))
        capacity_alpha = float(request.GET.get("capacity_alpha", 0.1))
        alpha_sigmoid_steepness = float(request.GET.get("alpha_sigmoid_steepness", 10.0))
        o_chigh = request.GET.get("override_consensus_high")
        o_clow = request.GET.get("override_consensus_low")

        override_consensus_high = float(o_chigh) if o_chigh else None
        override_consensus_low = float(o_clow) if o_clow else None

    except ValueError as e:
        return HttpResponse(f"Invalid parameter: {str(e)}", status=400)

    # 3) Rebuild the required objects
    sim_params = SimulationHyperparameters(
        kappa=kappa,
        bond_penalty=bond_penalty,
        liquid_alpha_consensus_mode=liquid_alpha_consensus_mode,
    )

    yuma_params = YumaParams(
        bond_moving_avg=bond_moving_avg,
        liquid_alpha=liquid_alpha,
        alpha_high=alpha_high,
        alpha_low=alpha_low,
        decay_rate=decay_rate,
        capacity_alpha=capacity_alpha,
        alpha_sigmoid_steepness=alpha_sigmoid_steepness,
        override_consensus_high=override_consensus_high,
        override_consensus_low=override_consensus_low,
    )

    # 4) Find the matching case
    package_cases = get_synthetic_cases(use_full_matrices=True, reset_bonds=reset_bonds)
    chosen_case = next((c for c in package_cases if c.name == case_name), None)
    if not chosen_case:
        return HttpResponse(f"Case '{case_name}' not found.", status=404)

    # 5) Generate chart/table HTML for this single case
    try:
        # 'generate_chart_table' can take a list of cases, so pass [chosen_case].
        # We'll also pass the single yuma version
        selected_yumas = [(chosen_yuma, yuma_params)]
        partial_html = generate_chart_table(cases=[chosen_case], yuma_versions=selected_yumas, yuma_hyperparameters=sim_params)
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(traceback_str)  # Or use logging
        return HttpResponse(f"Error generating chart: {str(e)}", status=500)

    # 6) Return the partial HTML snippet
    return HttpResponse(partial_html.data if partial_html else "No data", status=200)
