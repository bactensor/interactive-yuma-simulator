from dataclasses import asdict

from crispy_forms.helper import FormHelper
from crispy_forms.layout import Div, Field, Fieldset, Layout
from django import forms
from yuma_simulation._internal.cases import cases
from yuma_simulation._internal.yumas import YumaSimulationNames

yumas_dict = asdict(YumaSimulationNames())

# do not enable user to pick alpha version - it happens with a checkbox boolean
del yumas_dict["YUMA3_LIQUID"]


# WARNING: Refactored to cripsy forms by AI - review and maybe refactor


class SelectionForm(forms.Form):
    selected_cases = forms.MultipleChoiceField(
        choices=[(c.name, c.name) for c in cases],
        required=True,
        label="Select Cases",
        widget=forms.SelectMultiple(
            attrs={
                "class": "selectpicker form-control",
                "multiple": "multiple",
                "data-live-search": "true",
                "data-actions-box": "true",  # adds “Select All” / “Deselect All”
                "data-selected-text-format": "count > 3",  # “3 of 18 selected” style
            }
        ),
    )
    use_metagraph = forms.BooleanField(
        required=False,
        initial=False,
        label="Metagraph Case",
        widget=forms.CheckboxInput(attrs={"id": "id_use_metagraph"}),
    )

    start_block = forms.IntegerField(
        required=False,
        label="Start Block",
        widget=forms.NumberInput(attrs={"class": "form-control", "id": "id_start_block"}),
    )
    epochs_num = forms.IntegerField(
        required=False,
        label="Number of Epochs",
        min_value=1,
        max_value=100,
        error_messages={
            "min_value": "You must run at least 1 epoch.",
            "max_value": "You can request at most 100 epochs.",
        },
        widget=forms.NumberInput(attrs={"class": "form-control", "id": "id_epochs_num"}),
    )
    netuid = forms.IntegerField(
        required=False,
        label="Subnet ID",
        widget=forms.NumberInput(attrs={"class": "form-control", "id": "id_netuid"}),
    )
    validators = forms.CharField(
        required=False,
        label="Validators IDs (comma-sep)",
        widget=forms.TextInput(attrs={"class": "form-control", "id": "id_validators"}),
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
                "",
                Div(
                    Field("selected_cases"),
                    css_class="mode-block",
                    id="block_standard",
                ),
                Div(
                    Field("use_metagraph"),
                    Div(
                        Field("start_block"),
                        Field("epochs_num"),
                        Field("netuid"),
                        Field("validators"),
                        css_class="ml-4",
                        id="metagraph_params",
                    ),
                    css_class="mode-block",
                    id="block_metagraph",
                ),
                Field("selected_yumas"),
            ),
        )


class SimulationHyperparametersForm(forms.Form):
    kappa = forms.FloatField(
        initial=32767,
        max_value=65535,
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "data-bs-toggle": "tooltip",
                "data-bs-trigger": "hover",
                "data-bs-placement": "top",
                "title": "The consensus majority ratio: The weights set by validators who have lower normalized stake than Kappa are not used in calculating consensus, which affects ranks, which affect incentives.",
                "data-bs-container": "body",
            }
        ),
        error_messages={"max_value": "Must be at most 65535."},
    )
    bond_penalty = forms.FloatField(
        initial=65535,
        max_value=65535,
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "data-bs-toggle": "tooltip",
                "data-bs-trigger": "hover",
                "data-bs-placement": "top",
                "title": "The magnitude of the penalty subtracted from weights for exceeding consensus, for a specific subnet.",
                "data-bs-container": "body",
            }
        ),
        error_messages={"max_value": "Must be at most 65535."},
    )
    reset_bonds = forms.BooleanField(required=False, initial=False, label="Enable Reset Bonds")
    liquid_alpha_consensus_mode = forms.ChoiceField(
        choices=[("CURRENT", "Current"), ("PREVIOUS", "Previous"), ("MIXED", "Mixed")],
        initial="CURRENT",
        label="Liquid Alpha Consensus Mode",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.helper = FormHelper(self)
        self.helper.form_tag = False  # we’ll render this with the main form
        self.helper.disable_csrf = True
        self.helper.layout = Layout(
            Fieldset(
                "",  # empty heading - set in template
                # each field on its own row
                Div(
                    Field("kappa", wrapper_class="col-md-6 mb-3"),
                    Field("bond_penalty", wrapper_class="col-md-6 mb-3"),
                    css_class="row",
                ),
                Div(
                    Field("reset_bonds", css_class="mb-3"),
                    css_class="row",
                    id="row_reset_bonds",
                ),
                Div(Field("liquid_alpha_consensus_mode", css_class="mb-3"), css_class="row"),
            )
        )


class YumaParamsForm(forms.Form):
    bond_moving_avg = forms.FloatField(
        initial=900_000,
        max_value=1_000_000,
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "data-bs-toggle": "tooltip",
                "data-bs-trigger": "hover",
                "data-bs-placement": "top",
                "title": "The higher the bond moving average, the greater the influence of previously recorded bond values",
                "data-bs-container": "body",
            }
        ),
        error_messages={"max_value": "Must be at most 1 000 000."},
    )
    liquid_alpha = forms.BooleanField(required=False, initial=False)
    alpha_high = forms.FloatField(
        initial=0.3,
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "data-bs-toggle": "tooltip",
                "data-bs-trigger": "hover",
                "data-bs-placement": "top",
                "title": "The aggressive bound on liquid alpha. When buy/sell signals are strong (bold, away from consensus), the computed alpha moves toward this value to speed up bond acquisition or liquidation.",
                "data-bs-container": "body",
            }
        ),
    )
    alpha_low = forms.FloatField(
        initial=0.1,
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "data-bs-toggle": "tooltip",
                "data-bs-trigger": "hover",
                "data-bs-placement": "top",
                "title": "The conservative bound on liquid alpha. When buy/sell signals are weak (modest, close to consensus), the computed alpha moves toward this value to slow down bond adjustments.",
                "data-bs-container": "body",
            }
        ),
    )
    decay_rate = forms.FloatField(
        initial=0.1,
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "data-bs-toggle": "tooltip",
                "data-bs-trigger": "hover",
                "data-bs-placement": "top",
                "title": "Ensures that bonds associated with unsupported servers decrease over time.",
                "data-bs-container": "body",
            }
        ),
    )
    capacity_alpha = forms.FloatField(
        initial=0.1,
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "data-bs-toggle": "tooltip",
                "data-bs-trigger": "hover",
                "data-bs-placement": "top",
                "title": "Limits the bond purchase power per epoch.",
                "data-bs-container": "body",
            }
        ),
    )
    alpha_sigmoid_steepness = forms.FloatField(
        initial=10.0,
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "data-bs-toggle": "tooltip",
                "data-bs-trigger": "hover",
                "data-bs-placement": "top",
                "title": "A larger value makes the system snap quickly between conservative and aggressive bounds of liquid alpha as discrepancies change, while a smaller value yields a more gradual transition.",
                "data-bs-container": "body",
            }
        ),
    )

    def __init__(self, *args, selected_yuma=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.helper = FormHelper(self)
        self.helper.form_tag = False
        self.helper.disable_csrf = True
        self.helper.layout = Layout(
            Fieldset(
                "",  # empty heading - set in template
                # bond_moving_avg & liquid_alpha side-by-side
                Div(
                    Field("bond_moving_avg", wrapper_class="col-12 mb-3"),
                    Field("liquid_alpha", wrapper_class="col-12 mb-3"),
                    css_class="row bond-liquid-group",
                ),
                # alpha_high & alpha_low side-by-side (shown when liquid_alpha is checked)
                Div(
                    Field("alpha_high", wrapper_class="col-md-6 mb-3"),
                    Field("alpha_low", wrapper_class="col-md-6 mb-3"),
                    css_class="row alpha-params-group",
                ),
                # alpha_sigmoid_steepness on its own row
                Div(Field("alpha_sigmoid_steepness", wrapper_class="col-12 mb-3"), css_class="row alpha-params-group"),
                # decay_rate & capacity_alpha for special Yuma key
                Div(
                    Field("decay_rate", wrapper_class="col-md-6 mb-3"),
                    Field("capacity_alpha", wrapper_class="col-md-6 mb-3"),
                    css_class="row decay-capacity-group",
                ),
                # override fields are omitted entirely from the layout
            )
        )
        self.selected_yuma = selected_yuma
        if selected_yuma == 'YUMA1':
            f = self.fields['bond_moving_avg']
            f.required = False
            self.initial['bond_moving_avg'] = 0.0
    
    def clean_bond_moving_avg(self):
        val = self.cleaned_data.get('bond_moving_avg')
        if self.selected_yuma == 'YUMA1':
            return 0.0
        return val
