from dataclasses import asdict

from django import forms
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Field, Fieldset, Row, Column, Submit, Div
from crispy_forms.bootstrap import InlineCheckboxes

from yuma_simulation._internal.cases import cases
from yuma_simulation._internal.yumas import YumaSimulationNames

yumas_dict = asdict(YumaSimulationNames())
# hide alpha version behind a boolean toggle
del yumas_dict["YUMA_LIQUID"]


# WRANING: Rrefactored to cripsy forms by AI - review and maybe refactor

class SelectionForm(forms.Form):
    selected_cases = forms.MultipleChoiceField(
        choices=[(c.name, c.name) for c in cases],
        required=True,
        label="Select Cases",
        widget=forms.SelectMultiple(attrs={
            "class": "selectpicker form-control",
            "multiple": "multiple",
            "data-live-search": "true",
            "data-actions-box": "true",             # adds “Select All” / “Deselect All”
            "data-selected-text-format": "count > 3" # “3 of 18 selected” style
        }),
    )
    selected_yumas = forms.ChoiceField(
        choices=[(k, v) for k, v in yumas_dict.items()],
        label="Select Yuma Version",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.form_tag = False
        self.helper.disable_csrf = True
        self.helper.layout = Layout(
            Fieldset(
                "Simulation Configuration",
                # use Field instead of InlineCheckboxes
                Field("selected_cases"),
                Field("selected_yumas"),
            ),
        )


class SimulationHyperparametersForm(forms.Form):
    kappa = forms.FloatField(initial=0.5)
    bond_penalty = forms.FloatField(initial=1.0)
    reset_bonds = forms.BooleanField(required=False, initial=False,
                                     label="Enable Reset Bonds")
    liquid_alpha_consensus_mode = forms.ChoiceField(
        choices=[("CURRENT","Current"),("PREVIOUS","Previous"),("MIXED","Mixed")],
        initial="CURRENT",
        label="Liquid Alpha Consensus Mode",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.helper = FormHelper(self)
        self.helper.form_tag = False   # we’ll render this with the main form
        self.helper.disable_csrf = True
        self.helper.layout = Layout(
          Fieldset(
            "Hyperparameters",
            Row(
              Column('kappa', css_class="col-md-6"),
              Column('bond_penalty', css_class="col-md-6"),
            ),
            Row(
              Column('reset_bonds', css_class="col-md-6"),
              Column('liquid_alpha_consensus_mode', css_class="col-md-6"),
            ),
          )
        )


class YumaParamsForm(forms.Form):
    bond_moving_avg = forms.FloatField(initial=0.975)
    liquid_alpha     = forms.BooleanField(required=False, initial=False)
    alpha_high       = forms.FloatField(initial=0.3)
    alpha_low        = forms.FloatField(initial=0.1)
    decay_rate       = forms.FloatField(initial=0.1)
    capacity_alpha   = forms.FloatField(initial=0.1)
    alpha_sigmoid_steepness = forms.FloatField(initial=10.0)
    override_consensus_high  = forms.FloatField(required=False)
    override_consensus_low   = forms.FloatField(required=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.helper = FormHelper(self)
        self.helper.form_tag = False
        self.helper.disable_csrf = True
        self.helper.layout = Layout(
          Fieldset(
            "Yuma Parameters",
            Row(
              Column('bond_moving_avg', css_class="col-md-4"),
              Column('alpha_high', css_class="col-md-4"),
              Column('alpha_low', css_class="col-md-4"),
            ),
            Row(
              Column('decay_rate', css_class="col-md-4"),
              Column('capacity_alpha', css_class="col-md-4"),
              Column('alpha_sigmoid_steepness', css_class="col-md-4"),
            ),
            Row(
              Column('liquid_alpha', css_class="col-md-4"),
              Column('override_consensus_high', css_class="col-md-4"),
              Column('override_consensus_low', css_class="col-md-4"),
            ),
          )
        )
