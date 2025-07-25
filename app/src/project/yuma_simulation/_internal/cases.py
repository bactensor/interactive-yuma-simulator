import torch
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple
import logging
from .metagraph_utils import (
    slot_count, build_S_tensor, build_W_tensor,
    pick_validators, run_block_diagnostics,
)
import random
from django.conf import settings



logger = logging.getLogger(__name__)

class_registry = {}

def register_case(name: str):
    def decorator(cls):
        class_registry[name] = cls
        return cls

    return decorator

@dataclass
class BaseCase:
    base_validator: str
    name: str
    validators: list[str]
    num_epochs: int = 40
    reset_bonds: bool = False
    reset_bonds_index: int = None
    reset_bonds_epoch: int = None
    servers: list[str] = field(default_factory=lambda: ["Server 1", "Server 2"])
    use_full_matrices: bool = False
    chart_types: list[str] = field(default_factory=lambda: ["weights", "dividends", "bonds", "normalized_bonds"])

    @property
    def weights_epochs(self) -> list[torch.Tensor]:
        base_weights = self._get_base_weights_epochs
        if self.use_full_matrices:
            return [self.build_full_weights(W) for W in base_weights]
        return base_weights

    @property
    def _get_base_weights_epochs(self) -> list[torch.Tensor]:
        raise NotImplementedError("Subclasses must implement _get_base_weights_epochs().")


    @property
    def stakes_epochs(self) -> list[torch.Tensor]:
        base_stakes = self._get_base_stakes_epochs
        if self.use_full_matrices:
            return [self.build_full_stakes(S) for S in base_stakes]
        return base_stakes

    @property
    def _get_base_stakes_epochs(self) -> list[torch.Tensor]:
        return [torch.tensor([0.8, 0.1, 0.1])] * self.num_epochs

    def build_full_weights(self, W_base: torch.Tensor) -> torch.Tensor:
        """
        Given a weight matrix for validators (of shape [n_validators, n_servers]),
        embed it into a full square matrix of shape
          (n_validators + n_servers) x (n_validators + n_servers)
        so that the validators (rows) vote only in the server columns.
        """
        n_validators = len(self.validators)
        n_servers = len(self.servers)
        total = n_validators + n_servers
        W_full = torch.zeros(total, total)
        # Place the base matrix in the upper-right block.
        W_full[:n_validators, n_validators:] = W_base
        return W_full

    def build_full_stakes(self, stakes_valid: torch.Tensor) -> torch.Tensor:
        """
        Given a stakes vector for validators (of shape [n_validators]),
        append zeros for the servers (miners) so that the resulting tensor has shape
          (n_validators + n_servers,)
        """
        n_servers = len(self.servers)
        zeros = torch.zeros(n_servers, dtype=stakes_valid.dtype)
        return torch.cat([stakes_valid, zeros])

    def get_config_overrides(self) -> dict[str, Any]:
        return {}

    def __post_init__(self):
        if self.base_validator not in self.validators:
            raise ValueError(
                f"base_validator '{self.base_validator}' must be in validators list."
            )


@dataclass
class MetagraphCase(BaseCase):
    """
    A 'Case' that, for each metagraph (epoch), filters validators with S >= 1000 and
    provides weights and stakes only for those validators.
    """

    introduce_shift: bool = False
    shift_validator_hotkey: str = ""
    base_validator: str = ""
    num_epochs: int = 40

    name: str = "Dynamic Metagraph Case"
    metas: list[dict] = field(default_factory=list)  # List of metagraph dicts: { "S": ..., "W": ..., "hotkeys": ... }

    validators: list[str] = field(default_factory=list)
    requested_validators: list[str] = field(default_factory=list)

    # These will store per-epoch filtering information.
    valid_indices_epochs: list[list[int]] = field(default_factory=list, init=False)
    miner_indices_epochs: list[list[int]] = field(default_factory=list, init=False)
    validators_epochs: list[list[str]] = field(default_factory=list, init=False)
    servers: list[list[str]] = field(default_factory=list, init=False)

    hotkey_label_map: dict[str, str] = field(default_factory=dict, init=False)
    selected_servers: list[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        """
        Process each metagraph to compute epoch-specific valid indices, miner indices,
        and the corresponding validators (using the "hotkeys" field). The first epoch is
        then used to set parameters such as base_validator and top_validators.
        """
        if len(self.metas) == 0:
            raise ValueError("No metagraphs provided.")

        # Ensure all metas have torch.Tensor values for "S" and "W"
        for i, meta in enumerate(self.metas):
            if "S" in meta and not isinstance(meta["S"], torch.Tensor):
                meta["S"] = torch.tensor(meta["S"])
            if "W" in meta and not isinstance(meta["W"], torch.Tensor):
                meta["W"] = torch.tensor(meta["W"])

        if self.introduce_shift:
            self.name += " - shifted"

        # For each metagraph (epoch), compute the validators and miner indices.
        for idx, meta in enumerate(self.metas):
            stakes_tensor = meta["S"]  # shape [n_validators]
            mask = stakes_tensor >= 1000

            valid_indices = mask.nonzero(as_tuple=True)[0].tolist()

            n = stakes_tensor.size(0)
            miner_indices = list(range(n))

            if not valid_indices:
                raise ValueError(f"No validators have S >= 1000 in metagraph (epoch) {idx}.")

            self.valid_indices_epochs.append(valid_indices)
            self.miner_indices_epochs.append(miner_indices)
            # Get the list of validator and miners hotkeys for this epoch.
            try:
                validators_for_epoch = [meta["hotkeys"][uid] for uid in valid_indices]
                miners_for_epoch = [meta["hotkeys"][uid] for uid in miner_indices]
            except (KeyError, IndexError) as e:
                raise ValueError(f"Error retrieving hotkeys for epoch {idx}: {e}")

            self.validators_epochs.append(validators_for_epoch)
            self.servers.append(miners_for_epoch)

        #TODO (somehow refactor to not rely on base case post init logic just to satisfy it)
        if not self.base_validator:
            self.base_validator = self.requested_validators[0]
        self.validators = self.requested_validators

        super().__post_init__()

    @classmethod
    def from_mg_dumper_data(
        cls,
        mg_data: Dict[str, Any],
        requested_miners: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple["MetagraphCase", List[str], List[str]]:

        blocks          = mg_data["blocks"]
        hotkeys_by_blk  = {int(k): v for k, v in mg_data["hotkeys"].items()}
        weights         = mg_data["weights"]
        stakes          = mg_data["stakes"]
        n_slots         = slot_count(hotkeys_by_blk)

        requested_validators = pick_validators(hotkeys_by_blk=hotkeys_by_blk, stakes=stakes)

        # optional miner filter
        invalid_miners: List[str] = []
        if requested_miners:
            all_hks = set().union(*(set(h) for h in hotkeys_by_blk.values()))
            invalid_miners   = [hk for hk in requested_miners if hk not in all_hks]
            requested_miners = [hk for hk in requested_miners if hk in all_hks]

        # choose diagnostics blocks
        diag_blocks = set(random.sample(blocks, min(10, len(blocks))))
        diagnostics_enabled = getattr(settings, "ENABLE_METAGRAPH_DIAGNOSTICS", False)

        metas: List[Dict[str, Any]] = []
        for block in blocks:
            S = build_S_tensor(stakes[str(block)], n_slots)
            W = build_W_tensor(weights[str(block)], n_slots)

            slot_view = hotkeys_by_blk[block]

            # build is_active tensor for each uid
            is_active_t = torch.tensor(
                [tpl[2] for tpl in slot_view],
                dtype=torch.float32,
                device=S.device,
            )

            # zero stake for inactive accounts
            S *= is_active_t

            hk = [t[0] for t in slot_view if t]

            # comparing the fetched dumper data with on-chain data for testing purposes
            if diagnostics_enabled and block in diag_blocks:
                run_block_diagnostics(block, mg_data["netuid"], S, W, hk)

            metas.append({"S": S, "W": W, "hotkeys": hk})

        case = cls(
            metas=metas,
            num_epochs=len(metas),
            requested_validators=requested_validators,
            **kwargs,
        )
        case.hotkey_label_map = mg_data["labels"]
        case.selected_servers = requested_miners or []

        return case, invalid_miners
    
    @property
    def weights_epochs(self) -> list[torch.Tensor]:
        """
        Return a list of weight matrices (one per epoch) that have been filtered according
        to that epoch's valid (validators) and miner indices. If introduce_shift is enabled,
        then for each epoch (where possible) the row corresponding to shift_validator_id is
        replaced by the corresponding row values from the subsequent epoch, but only for
        miner columns that are common between the two epochs.
        """
        Ws = []
        for i, meta in enumerate(self.metas):
            W_full = meta["W"]
            valid_indices = self.valid_indices_epochs[i]
            miner_indices = self.miner_indices_epochs[i]
            # Filter rows (validators) and columns (miners)
            W_valid = W_full[valid_indices, :][:, miner_indices]
            Ws.append(W_valid)

        if not self.introduce_shift:
            return Ws

        # Apply shifting across epochs.
        for e in range(len(Ws) - 1):
            valid_indices_current = self.valid_indices_epochs[e]
            valid_indices_next = self.valid_indices_epochs[e + 1]

            # Get the hotkeys for current and next epochs
            hotkeys_current = [self.metas[e]["hotkeys"][idx] for idx in valid_indices_current]
            hotkeys_next = [self.metas[e + 1]["hotkeys"][idx] for idx in valid_indices_next]

            if (self.shift_validator_hotkey in hotkeys_current and
                    self.shift_validator_hotkey in hotkeys_next):
                row_current = hotkeys_current.index(self.shift_validator_hotkey)
                row_next = hotkeys_next.index(self.shift_validator_hotkey)

                miner_indices_current = self.miner_indices_epochs[e]
                miner_indices_next = self.miner_indices_epochs[e + 1]

                common_miners = set(miner_indices_current).intersection(miner_indices_next)
                for miner in common_miners:
                    col_current = miner_indices_current.index(miner)
                    col_next = miner_indices_next.index(miner)
                    Ws[e][row_current, col_current] = Ws[e + 1][row_next, col_next]
        return Ws

    @property
    def stakes_epochs(self) -> list[torch.Tensor]:
        """
        Return a list of stakes tensors (one per epoch) filtered to include only the valid
        validators as determined for each epoch.
        """
        Ss = []
        for i, meta in enumerate(self.metas):
            S_full = meta["S"]
            valid_indices = self.valid_indices_epochs[i]
            S_valid = S_full[valid_indices]
            Ss.append(S_valid)
        return Ss

    @property
    def stakes_dataframe(self) -> pd.DataFrame:
        """
        Convert the per-epoch stakes (torch.Tensors) into a DataFrame.
        Each row corresponds to an epoch and each column to a validator hotkey.
        Stakes for each epoch are normalized so that they sum to 1.
        Missing values (when a validator is not present in an epoch) will be NaN.
        """
        stakes_dict_list = []
        for epoch, stakes_tensor in enumerate(self.stakes_epochs):
            validators = self.validators_epochs[epoch]
            stakes_list = stakes_tensor.tolist()
            stakes_dict = {validator: stake for validator, stake in zip(validators, stakes_list)}
            stakes_dict_list.append(stakes_dict)

        df_stakes = pd.DataFrame(stakes_dict_list)
        df_stakes.index.name = "epoch"
        df_stakes = df_stakes.div(df_stakes.sum(axis=1), axis=0)
        return df_stakes


def create_case(case_name: str, **kwargs) -> BaseCase:
    if case_name not in class_registry:
        raise ValueError(f"Case '{case_name}' is not registered.")
    case_class = class_registry[case_name]
    return case_class(**kwargs)


@register_case("Case 1")
@dataclass
class Case1(BaseCase):
    name: str = "Case 1 - kappa moves first"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. (0.8)",
            "Small lazy vali. (0.1)",
            "Small lazier vali. (0.1)",
        ]
    )
    base_validator: str = "Big vali. (0.8)"

    @property
    def _get_base_weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_1 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch == 0:
                # Initially, consensus is achieved by all Validators
                W[:, 0] = 1.0
            elif epoch == 1:
                W[0, 1] = 1.0  # Validator A -> Server 2
                W[1, 0] = 1.0  # Validator B -> Server 1
                W[2, 0] = 1.0  # Validator C -> Server 1
            elif epoch == 2:
                W[0, 1] = 1.0  # Validator A -> Server 2
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 0] = 1.0  # Validator C -> Server 1
            else:
                # Subsequent epochs
                W[:, 1] = 1.0  # All validators -> Server 2
            weights_epochs_case_1.append(W)
        return weights_epochs_case_1


@register_case("Case 2")
@dataclass
class Case2(BaseCase):
    name: str = "Case 2 - kappa moves second"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. (0.8)",
            "Small eager vali. (0.1)",
            "Small lazy vali. (0.1)",
        ]
    )
    base_validator: str = "Small eager vali. (0.1)"

    @property
    def _get_base_weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_2 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch == 0:
                # Initially, consensus is achieved by all Validators
                W[:, 0] = 1.0
            elif epoch == 1:
                W[0, 0] = 1.0  # Validator A -> Server 1
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 0] = 1.0  # Validator C -> Server 1
            elif epoch == 2:
                W[0, 1] = 1.0  # Validator A -> Server 2
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 0] = 1.0  # Validator C -> Server 1
            else:
                # Subsequent epochs
                W[:, 1] = 1.0  # All validators -> Server 2
            weights_epochs_case_2.append(W)
        return weights_epochs_case_2


@register_case("Case 3")
@dataclass
class Case3(BaseCase):
    name: str = "Case 3 - kappa moves third"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. (0.8)",
            "Small eager vali. (0.1)",
            "Small lazy vali. (0.1)",
        ]
    )
    base_validator: str = "Small eager vali. (0.1)"

    @property
    def _get_base_weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_3 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch == 0:
                # Initially, consensus is achieved by all Validators
                W[:, 0] = 1.0
            elif epoch == 1:
                W[0, 0] = 1.0  # Validator A -> Server 1
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 0] = 1.0  # Validator C -> Server 1
            elif epoch == 2:
                W[0, 0] = 1.0  # Validator A -> Server 1
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 1] = 1.0  # Validator C -> Server 2
            else:
                # Subsequent epochs
                W[:, 1] = 1.0  # All validators -> Server 2
            weights_epochs_case_3.append(W)
        return weights_epochs_case_3


@register_case("Case 4")
@dataclass
class Case4(BaseCase):
    name: str = "Case 4 - all validators switch"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. (0.8)",
            "Small vali. (0.1)",
            "Small vali 2. (0.1)",
        ]
    )
    base_validator: str = "Big vali. (0.8)"

    @property
    def _get_base_weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_4 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch == 0:
                # All validators support Server 1
                W[0, 0] = 1.0  # Validator A -> Server 1
                W[1, 0] = 1.0  # Validator B -> Server 1
                W[2, 0] = 1.0  # Validator C -> Server 1
            if epoch >= 1:
                # All validators support Server 2
                W[0, 1] = 1.0  # Validator A -> Server 2
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 1] = 1.0  # Validator C -> Server 2
            weights_epochs_case_4.append(W)
        return weights_epochs_case_4


@register_case("Case 5")
@dataclass
class Case5(BaseCase):
    name: str = "Case 5 - kappa moves second, then third"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. (0.8)",
            "Small eager-eager vali. (0.1)",
            "Small eager-lazy vali. (0.1)",
        ]
    )
    base_validator: str = "Small eager-eager vali. (0.1)"
    reset_bonds: bool = True
    reset_bonds_index: int = 1
    reset_bonds_epoch: int = 20

    @property
    def _get_base_weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_5 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch == 0:
                # Initially, consensus is achieved by all Validators
                W[:, 0] = 1.0
            elif epoch == 1:
                W[0, 0] = 1.0  # Validator A -> Server 1
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 1] = 1.0  # Validator C -> Server 2
            elif epoch == 2:
                W[0, 1] = 1.0  # Validator A -> Server 2
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 1] = 1.0  # Validator C -> Server 2
            elif 3 <= epoch <= 20:
                # Subsequent epochs
                W[:, 1] = 1.0  # All validators -> Server 2
            elif epoch == 21:
                W[0, 1] = 1.0  # Validator A -> Server 2
                W[1, 0] = 1.0  # Validator B -> Server 1
                W[2, 1] = 1.0  # Validator C -> Server 2
            elif epoch == 22:
                W[0, 1] = 1.0  # Validator A -> Server 2
                W[1, 0] = 1.0  # Validator B -> Server 1
                W[2, 0] = 1.0  # Validator C -> Server 1
            else:
                # Subsequent epochs
                W[:, 0] = 1.0  # All validators -> Server 1
            weights_epochs_case_5.append(W)
        return weights_epochs_case_5


@register_case("Case 6")
@dataclass
class Case6(BaseCase):
    name: str = "Case 6 - kappa moves second, then all validators switch"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. (0.8)",
            "Small eager vali. (0.1)",
            "Small lazy vali. (0.1)",
        ]
    )
    base_validator: str = "Small eager vali. (0.1)"
    reset_bonds: bool = True
    reset_bonds_index: int = 0
    reset_bonds_epoch: int = 21

    @property
    def _get_base_weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_6 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch == 0:
                # All validators support Server 1
                W[:, 0] = 1.0
            elif epoch == 1:
                # Validator B switches to Server 2
                W[0, 0] = 1.0  # Validator A -> Server 1
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 0] = 1.0  # Validator C -> Server 1
            elif epoch == 2:
                W[0, 1] = 1.0  # Validator A -> Server 2
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 0] = 1.0  # Validator C -> Server 1
            elif 3 <= epoch <= 20:
                # All validators support Server 2
                W[:, 1] = 1.0
            else:
                # All validators switch back to Server 1
                W[:, 0] = 1.0
            weights_epochs_case_6.append(W)
        return weights_epochs_case_6


@register_case("Case 7")
@dataclass
class Case7(BaseCase):
    name: str = "Case 7 - big vali moves late, then all but one small vali moves late"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. (0.8)",
            "Small eager-lazy vali. (0.1)",
            "Small eager-eager vali. (0.1)",
        ]
    )
    base_validator: str = "Small eager-eager vali. (0.1)"
    reset_bonds: bool = True
    reset_bonds_index: int = 0
    reset_bonds_epoch: int = 21

    @property
    def _get_base_weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_7 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch == 0:
                # Initially, consensus is achieved by all Validators
                W[:, 0] = 1.0
            elif epoch == 1:
                W[0, 0] = 1.0  # Validator A -> Server 1
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 1] = 1.0  # Validator C -> Server 2
            elif epoch == 2:
                W[0, 1] = 1.0  # Validator A -> Server 2
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 1] = 1.0  # Validator C -> Server 2
            elif 3 <= epoch <= 20:
                # Subsequent epochs
                W[:, 1] = 1.0  # All validators -> Server 2
            elif epoch == 21:
                W[0, 1] = 1.0  # Validator A -> Server 1
                W[1, 1] = 1.0  # Validator B -> Server 1
                W[2, 0] = 1.0  # Validator C -> Server 2
            else:
                # Subsequent epochs
                W[:, 0] = 1.0  # All validators -> Server 1
            weights_epochs_case_7.append(W)
        return weights_epochs_case_7


@register_case("Case 8")
@dataclass
class Case8(BaseCase):
    name: str = "Case 8 - big vali moves late, then late"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big dishonest lazy vali. (0.8)",
            "Small eager-eager vali. (0.1)",
            "Small eager-eager vali 2. (0.1)",
        ]
    )
    base_validator: str = "Small eager-eager vali. (0.1)"
    reset_bonds: bool = True
    reset_bonds_index: int = 1
    reset_bonds_epoch: int = 20

    @property
    def _get_base_weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_8 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch == 0:
                W[:, 0] = 1.0
            elif epoch == 1:
                # Validators B and C switch to Server 2
                W[0, 0] = 1.0  # Validator A
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 1] = 1.0  # Validator C -> Server 2
            elif 2 <= epoch <= 20:
                # Validator A copies weights but still supports Server 1 with minimal weight
                W[0, 1] = 1.0  # Validator A
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 1] = 1.0  # Validator C -> Server 2
            elif epoch == 21:
                # Validators B and C switch back to Server 1
                W[0, 1] = 1.0  # Validator A
                W[1, 0] = 1.0  # Validator B -> Server 1
                W[2, 0] = 1.0  # Validator C -> Server 1
            else:
                W[:, 0] = 1.0
            weights_epochs_case_8.append(W)
        return weights_epochs_case_8


@register_case("Case 9")
@dataclass
class Case9(BaseCase):
    name: str = "Case 9 - small validators merged in e5"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. (0.8)",
            "Small vali. (0.1/0.2)",
            "Small vali 2. (0.1/0.0)",
        ]
    )
    base_validator: str = "Big vali. (0.8)"

    @property
    def _get_base_weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_9 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            W[:, 1] = 1.0  # All validators -> Server 2
            weights_epochs_case_9.append(W)
        return weights_epochs_case_9

    @property
    def _get_base_stakes_epochs(self) -> list[torch.Tensor]:
        stakes_epochs_case_9 = []
        for epoch in range(self.num_epochs):
            if 0 <= epoch <= 5:
                stakes = torch.tensor([0.8, 0.1, 0.1])
            else:
                stakes = torch.tensor([0.8, 0.2, 0.0])  # Validator C joins Validator B
            stakes_epochs_case_9.append(stakes)
        return stakes_epochs_case_9


@register_case("Case 10")
@dataclass
class Case10(BaseCase):
    name: str = "Case 10 - kappa delayed"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big delayed vali. (0.8)",
            "Small eager vali. (0.1)",
            "Small lazy vali. (0.1)",
        ]
    )
    base_validator: str = "Small eager vali. (0.1)"

    @property
    def _get_base_weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_10 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch == 0:
                # Initially, consensus is achieved by all Validators
                W[:, 0] = 1.0
            elif 1 <= epoch < 10:
                W[0, 0] = 1.0  # Validator A -> Server 1
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 0] = 1.0  # Validator C -> Server 1
            elif epoch == 10:
                W[0, 1] = 1.0  # Validator A -> Server 2
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 0] = 1.0  # Validator C -> Server 1
            else:
                # Subsequent epochs
                W[:, 1] = 1.0  # All validators -> Server 2
            weights_epochs_case_10.append(W)
        return weights_epochs_case_10


@register_case("Case 11")
@dataclass
class Case11(BaseCase):
    name: str = "Case 11 - clipping demo"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. 1 (0.49)",
            "Big vali. 2 (0.49)",
            "Small vali. (0.02)",
        ]
    )
    base_validator: str = "Big vali. 1 (0.49)"
    chart_types: list[str] = field(default_factory=lambda: ["weights", "dividends", "bonds", "normalized_bonds", "incentives"])

    @property
    def _get_base_weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_11 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch < 20:
                # Server 1
                W[0, 0] = 0.3
                W[1, 0] = 0.6
                W[2, 0] = 0.61
                # Server 2
                W[0, 1] = 0.7
                W[1, 1] = 0.4
                W[2, 1] = 0.39
            else:
                # Server 1
                W[0, 0] = 0.3
                W[1, 0] = 0.6
                W[2, 0] = 0.3
                # Server 2
                W[0, 1] = 0.7
                W[1, 1] = 0.4
                W[2, 1] = 0.61
            weights_epochs_case_11.append(W)
        return weights_epochs_case_11

    @property
    def _get_base_stakes_epochs(self) -> list[torch.Tensor]:
        return [torch.tensor([0.49, 0.49, 0.02])] * self.num_epochs


@register_case("Case 12")
@dataclass
class Case12(BaseCase):
    name: str = "Case 12 - all validators switch, but small validator/s support alt miner with minimal weight"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. (0.8)",
            "Small dishonest vali. (0.1)",
            "Small vali. (0.1)",
        ]
    )
    base_validator: str = "Big vali. (0.8)"
    chart_types: list[str] = field(default_factory=lambda: ["weights", "dividends", "bonds", "normalized_bonds", "incentives"])
    reset_bonds: bool = True
    reset_bonds_index: int = 1
    reset_bonds_epoch: int = 20

    @property
    def _get_base_weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_12 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch == 0:
                # All Validators support server 1
                W[0, 0] = 1.0
                W[1, :] = torch.tensor(
                    [0.999, 0.001]
                )  # Small dishonest vali. shifts slightly to Server 2
                W[2, 0] = 1.0
            elif 1 <= epoch <= 20:
                # All Validators support server 2
                W[0, 1] = 1.0
                W[1, :] = torch.tensor(
                    [0.001, 0.999]
                )  # Small dishonest vali. shifts back to Server 2
                W[2, 1] = 1.0
            else:
                # All Validators support server 1
                W[0, 0] = 1.0
                W[1, :] = torch.tensor([0.999, 0.001])
                W[2, 0] = 1.0
            weights_epochs_case_12.append(W)
        return weights_epochs_case_12


@dataclass
@register_case("Case 13")
class Case13(BaseCase):
    name: str = (
        "Case 13 - Big vali supports server 2, small validator/s support server 1"
    )
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. (0.8)",
            "Small vali. (0.1)",
            "Small vali 2. (0.1)",
        ]
    )
    base_validator: str = "Big vali. (0.8)"
    reset_bonds: bool = True
    reset_bonds_index: int = 0
    reset_bonds_epoch: int = 20

    @property
    def _get_base_weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_13 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch <= 20:
                W[0, 1] = 1.0  # Big vali. supports Server 2
                W[1, :] = torch.tensor([0.5, 0.5])  # Small vali. supports Server 1
                W[2, 1] = 1.0  # Small vali 2. supports Server 2
            else:
                W[0, 1] = 1.0  # Big vali. continues to support Server 2
                W[1, :] = torch.tensor([0.5, 0.5])  # Small vali. supports Server 1
                W[2, :] = torch.tensor([0.5, 0.5])  # Small vali 2. supports Server 1
            weights_epochs_case_13.append(W)
        return weights_epochs_case_13


@dataclass
@register_case("Case 14")
class Case14(BaseCase):
    name: str = "Case 14 - All validators support Server 1, one of them switches to Server 2 for one epoch"
    validators: list[str] = field(
        default_factory=lambda: ["Vali. 1 (0.33)", "Vali. 2 (0.33)", "Vali. 3 (0.34)"]
    )
    base_validator: str = "Vali. 1 (0.33)"
    reset_bonds: bool = False

    @property
    def _get_base_weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_14 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch >= 0 and epoch < 20:
                # Consensus is achieved by all Validators
                W[:, 0] = 1.0
            elif epoch == 20:
                W[0, 0] = 1.0  # Validator 1 -> Server 1
                W[1, 0] = 1.0  # Validator 2 -> Server 1
                W[2, 1] = 1.0  # Validator 3 -> Server 2
            else:
                W[:, 0] = 1.0  # All validators -> Server 1
            weights_epochs_case_14.append(W)
        return weights_epochs_case_14

    @property
    def _get_base_stakes_epochs(self) -> list[torch.Tensor]:
        return [torch.tensor([0.33, 0.33, 0.34])] * self.num_epochs

@register_case("Case 15")
@dataclass
class Case15(BaseCase):
    name: str = "Case 15 - big vali moves second, stable miner gets 0.25"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. (0.8)",
            "Small eager vali. (0.1)",
            "Small lazy vali. (0.1)",
        ]
    )
    base_validator: str = "Small eager vali. (0.1)"
    servers: list[str] = field(default_factory=lambda: ["Server 1", "Server 2", "Server 3"])
    chart_types: list[str] = field(default_factory=lambda: ["weights_subplots", "dividends", "bonds", "normalized_bonds"])

    @property
    def _get_base_weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_15 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 3)
            if epoch == 0:
                W[:, :] = torch.tensor([0.75, 0, 0.25])
            elif epoch == 1:
                W[0, :] = torch.tensor([0.75, 0, 0.25])
                W[1, :] = torch.tensor([0, 0.75, 0.25])
                W[2, :] = torch.tensor([0.75, 0, 0.25])
            elif epoch == 2:
                W[0, :] = torch.tensor([0, 0.75, 0.25])
                W[1, :] = torch.tensor([0, 0.75, 0.25])
                W[2, :] = torch.tensor([0.75, 0, 0.25])
            else:
                # Subsequent epochs
                W[:, :] = torch.tensor([0, 0.75, 0.25])
            weights_epochs_case_15.append(W)
        return weights_epochs_case_15

@register_case("Case 16")
@dataclass
class Case16(BaseCase):
    name: str = "Case 16 - big vali moves second, stable miner gets 0.5"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. (0.8)",
            "Small eager vali. (0.1)",
            "Small lazy vali. (0.1)",
        ]
    )
    base_validator: str = "Small eager vali. (0.1)"
    servers: list[str] = field(default_factory=lambda: ["Server 1", "Server 2", "Server 3"])
    chart_types: list[str] = field(default_factory=lambda: ["weights_subplots", "dividends", "bonds", "normalized_bonds"])

    @property
    def _get_base_weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_16 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 3)
            if epoch == 0:
                W[:, :] = torch.tensor([0.5, 0, 0.5])
            elif epoch == 1:
                W[0, :] = torch.tensor([0.5, 0, 0.5])
                W[1, :] = torch.tensor([0, 0.5, 0.5])
                W[2, :] = torch.tensor([0.5, 0, 0.5])
            elif epoch == 2:
                W[0, :] = torch.tensor([0, 0.5, 0.5])
                W[1, :] = torch.tensor([0, 0.5, 0.5])
                W[2, :] = torch.tensor([0.5, 0, 0.5])
            else:
                # Subsequent epochs
                W[:, :] = torch.tensor([0, 0.5, 0.5])
            weights_epochs_case_16.append(W)
        return weights_epochs_case_16

@register_case("Case 17")
@dataclass
class Case17(BaseCase):
    name: str = "Case 17 - big vali moves second, stable miner gets 0.75"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. (0.8)",
            "Small eager vali. (0.1)",
            "Small lazy vali. (0.1)",
        ]
    )
    base_validator: str = "Small eager vali. (0.1)"
    servers: list[str] = field(default_factory=lambda: ["Server 1", "Server 2", "Server 3"])
    chart_types: list[str] = field(default_factory=lambda: ["weights_subplots", "dividends", "bonds", "normalized_bonds"])

    @property
    def _get_base_weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_17 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 3)
            if epoch == 0:
                W[:, :] = torch.tensor([0.25, 0, 0.75])
            elif epoch == 1:
                W[0, :] = torch.tensor([0.25, 0, 0.75])
                W[1, :] = torch.tensor([0, 0.25, 0.75])
                W[2, :] = torch.tensor([0.25, 0, 0.75])
            elif epoch == 2:
                W[0, :] = torch.tensor([0, 0.25, 0.75])
                W[1, :] = torch.tensor([0, 0.25, 0.75])
                W[2, :] = torch.tensor([0.25, 0, 0.75])
            else:
                # Subsequent epochs
                W[:, :] = torch.tensor([0, 0.25, 0.75])
            weights_epochs_case_17.append(W)
        return weights_epochs_case_17


@register_case("Case 18")
@dataclass
class Case18(BaseCase):
    name: str = "Case 18 - big vali moves second, stable miner gets 0.9"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. (0.8)",
            "Small eager vali. (0.1)",
            "Small lazy vali. (0.1)",
        ]
    )
    base_validator: str = "Small eager vali. (0.1)"
    servers: list[str] = field(default_factory=lambda: ["Server 1", "Server 2", "Server 3"])
    chart_types: list[str] = field(default_factory=lambda: ["weights_subplots", "dividends", "bonds", "normalized_bonds"])

    @property
    def _get_base_weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_18 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 3)
            if epoch == 0:
                W[:, :] = torch.tensor([0.1, 0, 0.9])
            elif epoch == 1:
                W[0, :] = torch.tensor([0.1, 0, 0.9])
                W[1, :] = torch.tensor([0, 0.1, 0.9])
                W[2, :] = torch.tensor([0.1, 0, 0.9])
            elif epoch == 2:
                W[0, :] = torch.tensor([0, 0.1, 0.9])
                W[1, :] = torch.tensor([0, 0.1, 0.9])
                W[2, :] = torch.tensor([0.1, 0, 0.9])
            else:
                # Subsequent epochs
                W[:, :] = torch.tensor([0, 0.1, 0.9])
            weights_epochs_case_18.append(W)
        return weights_epochs_case_18


def _get_shared_3server_weights_epochs_low_noise(num_epochs: int, seed: int = 42) -> list[torch.Tensor]:
    """Generate shared proportional weights for 3 servers"""
    weights_epochs = []

    for epoch in range(num_epochs):
        W = torch.zeros(3, 3)

        # Server performance: Server 3 > Server 2 > Server 1 (small differences)
        true_perf = torch.tensor([0.345, 0.35, 0.36])

        gen_0 = torch.Generator().manual_seed(seed + 0 * num_epochs + epoch)
        gen_1 = torch.Generator().manual_seed(seed + 1 * num_epochs + epoch)
        gen_2 = torch.Generator().manual_seed(seed + 2 * num_epochs + epoch)

        W[0, :] = true_perf + torch.normal(0.0, 0.010, (3,), generator=gen_0)
        W[1, :] = true_perf + torch.normal(0.0, 0.015, (3,), generator=gen_1)
        W[2, :] = true_perf + torch.normal(0.0, 0.015, (3,), generator=gen_2)

        # Clamp to [0, 1] and normalize to proportional weights
        W = torch.clamp(W, 0, 1)
        W = (W.T / (W.sum(dim=1) + 1e-6)).T

        weights_epochs.append(W)

    return weights_epochs


@register_case("Case 19")
@dataclass
class Case19(BaseCase):
    name: str = "Case 19 - 3 servers Client-side WTA low noise"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. 1 (low noise)",
            "Big vali. 2 (low noise)",
            "Big vali. 3 (low noise)",
        ]
    )
    base_validator: str = "Big vali. 1 (low noise)"
    servers: list[str] = field(default_factory=lambda: ["Server 1", "Server 2", "Server 3"])
    chart_types: list[str] = field(default_factory=lambda: ["weights_subplots", "dividends", "bonds", "normalized_bonds", "incentives"])

    def get_config_overrides(self) -> dict:
        return {"winner_takes_all": True}

    @property
    def _get_base_weights_epochs(self) -> list[torch.Tensor]:
        shared_weights =  _get_shared_3server_weights_epochs_low_noise(self.num_epochs)
        # Apply client-side winner-takes-all
        wta_weights = []
        for W in shared_weights:
            W_wta = torch.zeros_like(W)
            for i in range(W.shape[0]):  # For each validator
                max_idx = torch.argmax(W[i])
                W_wta[i, max_idx] = 1.0
            wta_weights.append(W_wta)

        return wta_weights

    @property
    def _get_base_stakes_epochs(self) -> list[torch.Tensor]:
        return [torch.tensor([0.34, 0.33, 0.33])] * self.num_epochs


@register_case("Case 20")
@dataclass
class Case20(BaseCase):
    name: str = "Case 20 - 3 servers Server-side WTA  low noise"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. 1 (low noise)",
            "Big vali. 2 (low noise)",
            "Big vali. 3 (low noise)",
        ]
    )
    base_validator: str = "Big vali. 1 (low noise)"
    servers: list[str] = field(default_factory=lambda: ["Server 1", "Server 2", "Server 3"])
    chart_types: list[str] = field(default_factory=lambda: ["weights_subplots", "dividends", "bonds", "normalized_bonds", "incentives"])

    def get_config_overrides(self) -> dict:
        return {"winner_takes_all": True}

    @property
    def _get_base_weights_epochs(self) -> list[torch.Tensor]:
        return _get_shared_3server_weights_epochs_low_noise(self.num_epochs)

    @property
    def _get_base_stakes_epochs(self) -> list[torch.Tensor]:
        return [torch.tensor([0.34, 0.33, 0.33])] * self.num_epochs


# Shared weights generator function
def _get_shared_weights_epochs(num_epochs: int, seed: int = 42) -> list[torch.Tensor]:
    """Generate shared weights for both WTA cases"""
    torch.manual_seed(seed)
    weights_epochs = []

    for epoch in range(num_epochs):
        W = torch.zeros(3, 2)

        # Server 1 is slightly better than Server 2
        # Big vali 1 (low noise) - can distinguish better
        gen_0 = torch.Generator().manual_seed(seed + 0 * num_epochs + epoch)
        gen_1 = torch.Generator().manual_seed(seed + 1 * num_epochs + epoch)
        gen_2 = torch.Generator().manual_seed(seed + 2 * num_epochs + epoch)
        gen_3 = torch.Generator().manual_seed(seed + 3 * num_epochs + epoch)
        gen_4 = torch.Generator().manual_seed(seed + 4 * num_epochs + epoch)
        gen_5 = torch.Generator().manual_seed(seed + 5 * num_epochs + epoch)

        true_perf_1, true_perf_2 = 0.6, 0.55
        W[0, 0] = true_perf_1 + torch.normal(0.0, 0.05, (1,), generator=gen_0).item()  # Low noise
        W[0, 1] = true_perf_2 + torch.normal(0.0, 0.05, (1,), generator=gen_1).item()

        # Big vali 2 (high noise) - harder to distinguish
        W[1, 0] = true_perf_1 + torch.normal(0.0, 0.15, (1,), generator=gen_2).item()  # High noise
        W[1, 1] = true_perf_2 + torch.normal(0.0, 0.15, (1,), generator=gen_3).item()

        # Small vali (high noise)
        W[2, 0] = true_perf_1 + torch.normal(0.0, 0.15, (1,), generator=gen_4).item()  # High noise
        W[2, 1] = true_perf_2 + torch.normal(0.0, 0.15, (1,), generator=gen_5).item()

        # Clamp to [0, 1] and normalize
        W = torch.clamp(W, 0, 1)
        W = (W.T / (W.sum(dim=1) + 1e-6)).T

        weights_epochs.append(W)

    return weights_epochs


@register_case("Case 21")
@dataclass
class Case21(BaseCase):
    name: str = "Case 21 - 2 servers Client-side WTA high noise"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. 1 (low noise)",
            "Big vali. 2 (high noise)",
            "Small vali. (high noise)",
        ]
    )
    base_validator: str = "Big vali. 1 (low noise)"
    chart_types: list[str] = field(default_factory=lambda: ["weights", "dividends", "bonds", "normalized_bonds", "incentives"])

    @property
    def _get_base_weights_epochs(self) -> list[torch.Tensor]:
        shared_weights = _get_shared_weights_epochs(self.num_epochs)

        # Apply client-side winner-takes-all
        wta_weights = []
        for W in shared_weights:
            W_wta = torch.zeros_like(W)
            for i in range(W.shape[0]):  # For each validator
                max_idx = torch.argmax(W[i])
                W_wta[i, max_idx] = 1.0
            wta_weights.append(W_wta)

        return wta_weights

    @property
    def _get_base_stakes_epochs(self) -> list[torch.Tensor]:
        return [torch.tensor([0.49, 0.49, 0.02])] * self.num_epochs


@register_case("Case 22")
@dataclass
class Case22(BaseCase):
    name: str = "Case 22 - 2 servers Server-side WTA high noise"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. 1 (low noise)",
            "Big vali. 2 (high noise)",
            "Small vali. (high noise)",
        ]
    )
    base_validator: str = "Big vali. 1 (low noise)"
    chart_types: list[str] = field(default_factory=lambda: ["weights", "dividends", "bonds", "normalized_bonds", "incentives"])

    @property
    def _get_base_weights_epochs(self) -> list[torch.Tensor]:
        # Use raw proportional weights - Yuma will apply WTA
        return _get_shared_weights_epochs(self.num_epochs)

    @property
    def _get_base_stakes_epochs(self) -> list[torch.Tensor]:
        return [torch.tensor([0.49, 0.49, 0.02])] * self.num_epochs

    def get_config_overrides(self) -> dict[str, Any]:
        return {
            "winner_takes_all": True,
        }


def _get_shared_3server_weights_epochs(num_epochs: int, seed: int = 42) -> list[torch.Tensor]:
    """Generate shared proportional weights for 3 servers"""
    weights_epochs = []

    for epoch in range(num_epochs):
        W = torch.zeros(3, 3)

        # Server performance: Server 3 > Server 2 > Server 1 (small differences)
        true_perf = torch.tensor([0.30, 0.35, 0.40])

        gen_0 = torch.Generator().manual_seed(seed + 0 * num_epochs + epoch)
        gen_1 = torch.Generator().manual_seed(seed + 1 * num_epochs + epoch)
        gen_2 = torch.Generator().manual_seed(seed + 2 * num_epochs + epoch)

        W[0, :] = true_perf + torch.normal(0.0, 0.05, (3,), generator=gen_0)
        W[1, :] = true_perf + torch.normal(0.0, 0.10, (3,), generator=gen_1)
        W[2, :] = true_perf + torch.normal(0.0, 0.10, (3,), generator=gen_2)

        # Clamp to [0, 1] and normalize to proportional weights
        W = torch.clamp(W, 0, 1)
        W = (W.T / (W.sum(dim=1) + 1e-6)).T

        weights_epochs.append(W)

    return weights_epochs


@register_case("Case 23")
@dataclass
class Case23(BaseCase):
    name: str = "Case 23 - 3 servers Client-side WTA high noise"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. 1 (low noise)",
            "Big vali. 2 (high noise)",
            "Big vali. 3 (high noise)",
        ]
    )
    base_validator: str = "Big vali. 1 (low noise)"
    servers: list[str] = field(default_factory=lambda: ["Server 1", "Server 2", "Server 3"])
    chart_types: list[str] = field(default_factory=lambda: ["weights_subplots", "dividends", "bonds", "normalized_bonds", "incentives"])

    def get_config_overrides(self) -> dict:
        return {"winner_takes_all": True}

    @property
    def _get_base_weights_epochs(self) -> list[torch.Tensor]:
        shared_weights =  _get_shared_3server_weights_epochs(self.num_epochs)
        # Apply client-side winner-takes-all
        wta_weights = []
        for W in shared_weights:
            W_wta = torch.zeros_like(W)
            for i in range(W.shape[0]):  # For each validator
                max_idx = torch.argmax(W[i])
                W_wta[i, max_idx] = 1.0
            wta_weights.append(W_wta)

        return wta_weights

    @property
    def _get_base_stakes_epochs(self) -> list[torch.Tensor]:
        return [torch.tensor([0.34, 0.33, 0.33])] * self.num_epochs


@register_case("Case 24")
@dataclass
class Case24(BaseCase):
    name: str = "Case 24 - 3 servers Server-side WTA high noise"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. 1 (low noise)",
            "Big vali. 2 (high noise)",
            "Big vali. 3 (high noise)",
        ]
    )
    base_validator: str = "Big vali. 1 (low noise)"
    servers: list[str] = field(default_factory=lambda: ["Server 1", "Server 2", "Server 3"])
    chart_types: list[str] = field(default_factory=lambda: ["weights_subplots", "dividends", "bonds", "normalized_bonds", "incentives"])

    def get_config_overrides(self) -> dict:
        return {"winner_takes_all": True}

    @property
    def _get_base_weights_epochs(self) -> list[torch.Tensor]:
        return _get_shared_3server_weights_epochs(self.num_epochs)

    @property
    def _get_base_stakes_epochs(self) -> list[torch.Tensor]:
        return [torch.tensor([0.34, 0.33, 0.33])] * self.num_epochs

cases = [cls() for case_name, cls in class_registry.items()]

def get_synthetic_cases(use_full_matrices: bool = False, reset_bonds: bool = False) -> list[BaseCase]:
    """
    Creates all synthetic cases.

    Parameters:
      use_full_matrices (bool): If True, uses full matrices as defined in the BaseCase (as implemented in real Rust Yuma).
      reset_bonds (bool): If False, forces reset_bonds to be off for every case,
                                 even if a case explicitly sets it to True.
                                 If True, only cases that specify reset_bonds=True will have it enabled.
    """
    cases = []
    for cls in class_registry.values():
        instance = cls(use_full_matrices=use_full_matrices)
        if not reset_bonds:
            instance.reset_bonds = False
        cases.append(instance)
    return cases
