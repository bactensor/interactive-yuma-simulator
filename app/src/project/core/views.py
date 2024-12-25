from dataclasses import asdict

from django.http import HttpResponse
from django.shortcuts import render
from yuma_simulation._internal.cases import cases as package_cases
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
        selected_case_names = selection_form.cleaned_data["selected_cases"]
        chosen_cases = [c for c in package_cases if c.name in selected_case_names]

        # Extract selected YUMAs and map back using their keys
        yumas_dict = asdict(YumaSimulationNames())
        selected_yuma_key = selection_form.cleaned_data["selected_yumas"]  # Single key
        selected_yuma = yumas_dict[selected_yuma_key]

        # Create SimulationHyperparameters object
        hyper_data = hyper_form.cleaned_data
        sim_params = SimulationHyperparameters(
            kappa=hyper_data["kappa"],
            bond_penalty=hyper_data["bond_penalty"],
        )

        # Create YumaParams object
        yuma_data = yuma_form.cleaned_data
        yuma_params_obj = YumaParams(
            bond_alpha=yuma_data["bond_alpha"],
            liquid_alpha=yuma_data["liquid_alpha"],
            alpha_high=yuma_data["alpha_high"],
            alpha_low=yuma_data["alpha_low"],
            decay_rate=yuma_data["decay_rate"],
            capacity_alpha=yuma_data["capacity_alpha"],
            override_consensus_high=yuma_data["override_consensus_high"],
            override_consensus_low=yuma_data["override_consensus_low"],
        )

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
