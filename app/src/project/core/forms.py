from dataclasses import asdict

from django import forms
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Field, Fieldset, Div
from crispy_forms.bootstrap import InlineCheckboxes
from crispy_bootstrap5.bootstrap5 import FloatingField

from yuma_simulation._internal.cases import cases
from yuma_simulation._internal.yumas import YumaSimulationNames

yumas_dict = asdict(YumaSimulationNames())

#do not enable user to pick alpha version - it happens with a checkbox boolean
del yumas_dict["YUMA3_LIQUID"]


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
                "", # empty heading - set in template
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
            "", # empty heading - set in template
            # each field on its own row
            Div(
              Field('kappa', wrapper_class='col-md-6 mb-3'),
              Field('bond_penalty', wrapper_class='col-md-6 mb-3'),
              css_class='row'
            ),
            Div(Field('reset_bonds',                css_class='mb-3'), css_class='row'),
            Div(Field('liquid_alpha_consensus_mode',css_class='mb-3'), css_class='row'),
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.helper = FormHelper(self)
        self.helper.form_tag = False
        self.helper.disable_csrf = True
        self.helper.layout = Layout(
          Fieldset(
            "", # empty heading - set in template

            # bond_moving_avg & liquid_alpha side-by-side
            Div(
              Field('bond_moving_avg', wrapper_class='col-12 mb-3'),
              Field('liquid_alpha',    wrapper_class='col-12 mb-3'),
              css_class='row bond-liquid-group'
            ),

            # alpha_high & alpha_low side-by-side (shown when liquid_alpha is checked)
            Div(
              Field('alpha_high', wrapper_class='col-md-6 mb-3'),
              Field('alpha_low',  wrapper_class='col-md-6 mb-3'),
              css_class='row alpha-params-group'
            ),

            # alpha_sigmoid_steepness on its own row
            Div(
              Field('alpha_sigmoid_steepness', wrapper_class='col-12 mb-3'),
              css_class='row alpha-params-group'
            ),

            # decay_rate & capacity_alpha for special Yuma key
            Div(
              Field('decay_rate',     wrapper_class='col-md-6 mb-3'),
              Field('capacity_alpha', wrapper_class='col-md-6 mb-3'),
              css_class='row decay-capacity-group'
            ),

            # override fields are omitted entirely from the layout
          )
        )