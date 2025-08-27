import logging
from datetime import datetime
from typing import Tuple, List

from project.core.utils import fetch_metagraph_weights_stakes, fetch_metagraph_rewards
from project.yuma_simulation._internal.cases import MetagraphCase

logger = logging.getLogger(__name__)


def prepare_metagraph_data(
    start_date: datetime,
    end_date: datetime,
    netuid: int,
    max_epochs: int = 2,
) -> tuple[MetagraphCase, List[int]]:
    """
    Fetch and prepare metagraph data for validation.

    Returns (MetagraphCase, blocks_tested).
    Raises ValueError if insufficient epochs are available.
    """
    logger.info(f"Fetching metagraph data for up to {max_epochs} epochs...")
    try:
        weights_stakes_data = fetch_metagraph_weights_stakes(
            start_date=start_date, end_date=end_date, netuid=netuid
        )
        blocks = weights_stakes_data.get("blocks", [])
        if len(blocks) < max_epochs:
            logger.warning(
                f"Only {len(blocks)} blocks available, requested {max_epochs}"
            )
            max_epochs = len(blocks)
        if max_epochs < 3:
            raise ValueError(
                f"Need at least 3 epochs for validation (epoch 0 for B_old, epochs 1+ for validation), got {max_epochs}"
            )

        rewards_data = fetch_metagraph_rewards(
            start_date=start_date, end_date=end_date, netuid=netuid
        )
        mg_data = {**weights_stakes_data, **rewards_data}
    except Exception as e:
        logger.error(f"Failed to fetch metagraph data: {e}")
        raise

    try:
        case, _ = MetagraphCase.from_mg_dumper_data(mg_data)
    except Exception as e:
        logger.error(f"Error creating MetagraphCase: {e}")
        raise

    if case.num_epochs < max_epochs:
        logger.info(f"Limiting epochs from {case.num_epochs} to {max_epochs}")
        max_epochs = case.num_epochs

    case.num_epochs = max_epochs
    case.metas = case.metas[:max_epochs]
    tested_blocks = blocks[:max_epochs]
    return case, tested_blocks

