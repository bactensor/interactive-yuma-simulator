from dataclasses import asdict

from django import forms
from yuma_simulation._internal.cases import cases
from yuma_simulation._internal.yumas import YumaSimulationNames

yumas_dict = asdict(YumaSimulationNames())

#do not enable user to pick alpha version - it happens with a checkbox boolean
del yumas_dict["YUMA_LIQUID"]
# del yumas_dict["YUMA4_LIQUID"]

class SimulationHyperparametersForm(forms.Form):
    kappa = forms.FloatField(initial=0.5)
    bond_penalty = forms.FloatField(initial=1.0)
    reset_bonds = forms.BooleanField(
        required=False, 
        initial=False, 
        label="Enable Reset Bonds"
    )
    liquid_alpha_consensus_mode = forms.ChoiceField(
        choices=[
            ("CURRENT", "CURRENT"),
            ("PREVIOUS", "PREVIOUS"),
            ("MIXED", "MIXED"),
        ],
        initial="CURRENT",
        label="Liquid Alpha Consensus Mode"
    )


class YumaParamsForm(forms.Form):
    bond_moving_avg = forms.FloatField(initial=0.975)
    liquid_alpha = forms.BooleanField(required=False, initial=False)
    alpha_high = forms.FloatField(initial=0.3)
    alpha_low = forms.FloatField(initial=0.1)
    decay_rate = forms.FloatField(initial=0.1)
    capacity_alpha = forms.FloatField(initial=0.1)
    alpha_sigmoid_steepness = forms.FloatField(initial=10.0)
    override_consensus_high = forms.FloatField(required=False)
    override_consensus_low = forms.FloatField(required=False)


class SelectionForm(forms.Form):
    selected_cases = forms.MultipleChoiceField(
        choices=[(c.name, c.name) for c in cases],
        widget=forms.CheckboxSelectMultiple,
        required=True,
        label="Select Cases",
    )
    selected_yumas = forms.ChoiceField(
        choices=[(key, value) for key, value in yumas_dict.items()],
        widget=forms.Select,
        required=True,
        label="Select Yuma Version",
    )
