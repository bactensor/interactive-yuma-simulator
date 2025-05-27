import json
import logging
import re
from dataclasses import asdict

import pandas as pd
from django.http import HttpResponse, HttpResponseServerError
from django.shortcuts import render
from django.views.decorators.cache import cache_control
from requests.exceptions import HTTPError, Timeout
from yuma_simulation._internal.cases import get_synthetic_cases, MetagraphCase
from yuma_simulation._internal.yumas import SimulationHyperparameters, YumaParams, YumaSimulationNames
from yuma_simulation.v1 import api as yuma_api
from yuma_simulation.v1.api import generate_chart_table, generate_metagraph_based_chart_table

from .forms import SelectionForm, SimulationHyperparametersForm, YumaParamsForm
from .utils import ONE_MILLION, UINT16_MAX, fetch_metagraph_data, normalize

logger = logging.getLogger(__name__)


@cache_control(public=True, max_age=604800, s_maxage=604800)
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

        if cleaned.get("use_metagraph", False):
            start = cleaned["start_block"]
            netuid = cleaned["netuid"]
            raw_validators = cleaned.get("validators", "")

            picked_validators = check_validators(
                selection_form,
                start_block=start,
                netuid=netuid,
                raw_validators=raw_validators,
            )

            if picked_validators is not None:
                cleaned["validators"] = picked_validators

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

            context["yumas_json"] = json.dumps(
                {
                    "selected_yuma_key": yuma_key,
                    "chosen_yuma": chosen_yuma,
                    "sim_params": sim_params,
                    "yuma_params": yuma_params,
                }
            )

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

@cache_control(public=True, max_age=604800, s_maxage=604800)
def metagraph_simulation_view(request):
    try:
        raw_kappa = float(request.GET.get("kappa", 32767))
        raw_bond_penalty = float(request.GET.get("bond_penalty", 65535))
        lam = request.GET.get("liquid_alpha_consensus_mode", "CURRENT")
        liq_alpha = request.GET.get("liquid_alpha", "False") == "true"
        chosen_yuma = request.GET.get("selected_yumas", "YUMA")

        raw_bma = float(request.GET.get("bond_moving_avg", 900_000))
        alpha_high = float(request.GET.get("alpha_high", 0.3))
        alpha_low = float(request.GET.get("alpha_low", 0.1))
        decay_rate = float(request.GET.get("decay_rate", 0.1))
        cap_alpha = float(request.GET.get("capacity_alpha", 0.1))
        steepness = float(request.GET.get("alpha_sigmoid_steepness", 10.0))

        start_block = int(request.GET.get("start_block", 0))
        epochs_num = int(request.GET.get("epochs_num", 0))
        netuid = int(request.GET.get("netuid", 0))

        raw_validators = request.GET.get("validators", "")
        validators = [int(v.strip()) for v in raw_validators.split(",") if v.strip()]

        raw_miners = request.GET.get("miners","")
        miners_ids = [int(m.strip()) for m in raw_miners.split(",") if m.strip()]

        raw_alpha_tao = float(request.GET.get("alpha_tao_ratio", 0.1))
        
    except ValueError as e:
        return HttpResponse(f"Invalid parameter: {e}", status=400)

    sim_params = SimulationHyperparameters(
        kappa=normalize(raw_kappa, UINT16_MAX),
        bond_penalty=normalize(raw_bond_penalty, UINT16_MAX),
        liquid_alpha_consensus_mode=lam,
        alpha_tao_ratio=raw_alpha_tao,
    )
    yuma_params = YumaParams(
        bond_moving_avg=normalize(raw_bma, ONE_MILLION),
        liquid_alpha=liq_alpha,
        alpha_high=alpha_high,
        alpha_low=alpha_low,
        decay_rate=decay_rate,
        capacity_alpha=cap_alpha,
        alpha_sigmoid_steepness=steepness,
    )

    end_block = start_block + epochs_num * 360
    try:
        mg_data = fetch_metagraph_data(
            start_block=start_block,
            end_block=end_block,
            netuid=netuid,
        )
    except Timeout:
        return HttpResponseServerError(
            "The service timed out. "
            "Try requesting fewer epochs."
        )
    except HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return HttpResponse(f"No metagraph data for blocks {start_block}–{end_block}", status=404)
        return HttpResponse(f"Error fetching metagraph data: {e}", status=400)
    except Exception as e:
        return HttpResponse(f"Error fetching metagraph data: {e}", status=400)

    try:
        case = MetagraphCase.from_mg_dumper_data(
            mg_data=mg_data,
            top_validators_ids=validators,
            netuid=netuid,
            selected_miners=miners_ids,
            )
    except ValueError as e:
        return HttpResponse(str(e), status=400)

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
        epochs_padding=0,
    )

    return HttpResponse(partial_html.data or "No data", status=200)


def check_validators(
    form,
    start_block: int,
    netuid: int,
    raw_validators: str,
    stake_threshold: int = 1000,
) -> list[int] | None:
    """
    Fetch metagraph data for `start_block`/`netuid`,
    validate the comma-separated IDs in `raw_validators`
    against the first‐epoch stakes, and add errors to `form`
    as needed. Returns a list of ints (possibly empty)
    or None if the fetch itself failed in a fatal way.
    """
    try:
        mg_data = fetch_metagraph_data(
            start_block=start_block,
            end_block=start_block,
            netuid=netuid,
        )
    except HTTPError as e:
        resp = e.response
        detail = None
        try:
            detail = resp.json().get("error")
        except ValueError:
            detail = resp.text[:200]
        if resp.status_code == 404:
            msg = f"No metagraph data for block {start_block}"
        else:
            msg = f"Error fetching metagraph data ({resp.status_code}): {detail}"
        form.add_error("start_block", msg)
        return None
    except Exception as e:
        form.add_error("start_block", f"Error fetching metagraph data: {e}")
        return None


    # extract uids & stakes
    uids = mg_data.get("uids", [])
    stakes = mg_data.get("stakes", {})
    if not stakes:
        form.add_error("start_block", "Metagraph returned no stakes data")
        return None

    first_epoch = next(iter(stakes))
    first_stakes = stakes[first_epoch]
    valid_idxs = {int(idx_str) for idx_str, stake in first_stakes.items() if stake > stake_threshold}
    valid_uids = {uids[i] for i in valid_idxs if i < len(uids)}

    # parse and validate the raw comma-list
    picked_validators: list[int] = []
    for tok in raw_validators.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vid = int(tok)
        except ValueError:
            form.add_error("validators", f"‘{tok}’ is not a valid integer")
            continue

        picked_validators.append(vid)
        if vid not in valid_uids:
            form.add_error("validators", f"Validator ID {vid} has zero stake in epoch {first_epoch}")

    return picked_validators


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
        raw_img_tag = summary_table.iat[i, c_idx]
        img_src = parse_img_src(raw_img_tag)

        rows_html.append(f'''
          <div class="mb-4 text-center">
            <img src="{img_src}"
                 class="img-fluid w-100"
                 alt="Chart for {case_name}">
          </div>
        ''')

    return f'''
    <div class="container-fluid px-3">
      {''.join(rows_html)}
    </div>
    '''


yuma_api._generate_ipynb_table = bootstrap_generate_ipynb_table