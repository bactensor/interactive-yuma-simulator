from dataclasses import asdict

from django.http import HttpResponse
from django.shortcuts import render
from yuma_simulation._internal.cases import get_synthetic_cases
from yuma_simulation._internal.yumas import SimulationHyperparameters, YumaParams, YumaSimulationNames
from yuma_simulation.v1.api import generate_chart_table

from .forms import SelectionForm, SimulationHyperparametersForm, YumaParamsForm


def simulation_view(request):
    # Initialize forms with either GET data or empty if not submitted
    selection_form = SelectionForm(request.GET or None)
    hyper_form = SimulationHyperparametersForm(request.GET or None)
    yuma_form = YumaParamsForm(request.GET or None)

    chart_table_html = None

    # Check if forms are valid after user submits (via GET)
    if selection_form.is_valid() and hyper_form.is_valid() and yuma_form.is_valid():
        # Extract selected cases and map them back to case objects by their names
        hyper_data = hyper_form.cleaned_data
        reset_bonds_value = hyper_data["reset_bonds"]
        package_cases = get_synthetic_cases(use_full_matrices=True, reset_bonds=reset_bonds_value)

        selected_case_names = selection_form.cleaned_data["selected_cases"]
        chosen_cases = [c for c in package_cases if c.name in selected_case_names]

        # Create SimulationHyperparameters object
        sim_params = SimulationHyperparameters(
            kappa=hyper_data["kappa"],
            bond_penalty=hyper_data["bond_penalty"],
            liquid_alpha_consensus_mode=hyper_data["liquid_alpha_consensus_mode"],
        )

        # Create YumaParams object
        yuma_data = yuma_form.cleaned_data
        yuma_params_obj = YumaParams(
            bond_moving_avg=yuma_data["bond_moving_avg"],
            liquid_alpha=yuma_data["liquid_alpha"],
            alpha_high=yuma_data["alpha_high"],
            alpha_low=yuma_data["alpha_low"],
            decay_rate=yuma_data["decay_rate"],
            capacity_alpha=yuma_data["capacity_alpha"],
            alpha_sigmoid_steepness=yuma_data["alpha_sigmoid_steepness"],
            override_consensus_high=yuma_data["override_consensus_high"],
            override_consensus_low=yuma_data["override_consensus_low"],
        )

        # Extract selected YUMAs and map back using their keys
        yumas_dict = asdict(YumaSimulationNames())
        selected_yuma_key = selection_form.cleaned_data["selected_yumas"]  # Single key

        if selected_yuma_key in ["YUMA", "YUMA4"] and yuma_data["liquid_alpha"] == True:
            selected_yuma_key += "_LIQUID"
            
        selected_yuma = yumas_dict[selected_yuma_key]


        # Build yuma_versions
        selected_yumas = [(selected_yuma, yuma_params_obj)]

        # Generate the chart table
        try:
            # Assume generate_chart_table returns a string containing HTML
            chart_table_html = generate_chart_table(chosen_cases, selected_yumas, sim_params)
        except Exception as e:
            return HttpResponse(f"Error: {e}", status=500)

    return render(
        request,
        "simulator.html",
        {
            "selection_form": selection_form,
            "hyper_form": hyper_form,
            "yuma_form": yuma_form,
            "chart_table_html": chart_table_html.data if chart_table_html else None,
        },
    )
