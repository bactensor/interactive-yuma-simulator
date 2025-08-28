import logging
from datetime import datetime
from typing import Tuple, List, Optional

from project.core.utils import fetch_metagraph_weights_stakes, fetch_metagraph_rewards
from project.yuma_simulation._internal.cases import MetagraphCase

logger = logging.getLogger(__name__)


def prepare_metagraph_data(
    *,
    netuid: int,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    start_block: Optional[int] = None,
    end_block: Optional[int] = None,
    num_epochs: int = 3,
) -> tuple[MetagraphCase, List[int]]:
    """
    Fetch and prepare metagraph data for validation.

    Returns (MetagraphCase, blocks_tested).
    Raises ValueError if insufficient epochs are available.
    """
    logger.info("Fetching metagraph data...")
    try:
        if num_epochs < 3:
            raise ValueError(
                f"Need at least 3 epochs for validation (epoch 0 for B_old, epochs 1+ for validation), got {num_epochs}"
            )

        weights_stakes_data = fetch_metagraph_weights_stakes(
            netuid=netuid,
            start_date=start_date,
            end_date=end_date,
            start_block=start_block,
            end_block=end_block,
            num_epochs=num_epochs,
        )
        blocks = weights_stakes_data.get("blocks", [])
        desired_epochs = num_epochs
        if len(blocks) < desired_epochs:
            logger.warning(
                f"Only {len(blocks)} blocks available, requested {desired_epochs}"
            )
            desired_epochs = len(blocks)
        if desired_epochs < 3:
            raise ValueError(
                f"Need at least 3 epochs for validation (epoch 0 for B_old, epochs 1+ for validation), got {desired_epochs}"
            )

        rewards_data = fetch_metagraph_rewards(
            netuid=netuid,
            start_date=start_date,
            end_date=end_date,
            start_block=start_block,
            end_block=end_block,
            num_epochs=desired_epochs,
        )
        mg_data = {**weights_stakes_data, **rewards_data}
        if "hotkeys" in weights_stakes_data:
            mg_data["hotkeys"] = weights_stakes_data["hotkeys"]
        if "labels" in weights_stakes_data:
            mg_data["labels"] = weights_stakes_data["labels"]
    except Exception as e:
        logger.error(f"Failed to fetch metagraph data: {e}")
        raise

    try:
        case, _ = MetagraphCase.from_mg_dumper_data(mg_data)
    except Exception as e:
        logger.error(f"Error creating MetagraphCase: {e}")
        raise

    # Harmonize number of epochs to compare
    effective_epochs = min(desired_epochs, case.num_epochs)
    if case.num_epochs != effective_epochs:
        logger.info(f"Limiting epochs from {case.num_epochs} to {effective_epochs}")

    case.num_epochs = effective_epochs
    case.metas = case.metas[:effective_epochs]
    tested_blocks = blocks[:effective_epochs]
    return case, tested_blocks
