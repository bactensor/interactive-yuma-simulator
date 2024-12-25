from django import forms
from yuma_simulation._internal.cases import cases
from yuma_simulation._internal.yumas import YumaSimulationNames
from dataclasses import asdict

yumas_dict = asdict(YumaSimulationNames())

class SimulationHyperparametersForm(forms.Form):
    kappa = forms.FloatField(initial=0.5)
    bond_penalty = forms.FloatField(initial=1.0)

class YumaParamsForm(forms.Form):
    # For simplicity, one YumaParamsForm. In reality, you might need one per selected Yuma.
    bond_alpha = forms.FloatField(initial=0.1)
    liquid_alpha = forms.BooleanField(required=False, initial=False)
    alpha_high = forms.FloatField(initial=0.9)
    alpha_low = forms.FloatField(initial=0.7)
    decay_rate = forms.FloatField(initial=0.1)
    capacity_alpha = forms.FloatField(initial=0.1)
    override_consensus_high = forms.FloatField(required=False)
    override_consensus_low = forms.FloatField(required=False)

class SelectionForm(forms.Form):
    selected_cases = forms.MultipleChoiceField(
        choices=[(c.name, c.name) for c in cases],
        widget=forms.CheckboxSelectMultiple,
        required=True,
        label="Select Cases"
    )
    selected_yumas = forms.ChoiceField(
        choices=[(key, value) for key, value in yumas_dict.items()],
        widget=forms.Select,
        required=True,
        label="Select Yuma Version"
    )
