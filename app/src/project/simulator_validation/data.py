import logging
from datetime import datetime, timedelta
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

        # Prefer start + num_epochs so backend can snap to epoch-start blocks.
        # If num_epochs is provided, omit explicit end parameters.
        original_end_date = end_date
        use_end_date = None if num_epochs is not None else end_date
        use_end_block = None if num_epochs is not None else end_block

        # Retry strategy: if backend reports "No epoch-start blocks", move start_date earlier
        # to allow it to find the nearest epoch boundary. Try a few backoff windows.
        backoff_hours = [0, 6, 12, 24, 48, 96]
        last_err: Exception | None = None
        chosen_start_date = start_date
        weights_stakes_data = None
        for bh in backoff_hours:
            try:
                attempt_start_date = (
                    (start_date - timedelta(hours=bh)) if start_date is not None else None
                )
                chosen_start_date = attempt_start_date
                weights_stakes_data = fetch_metagraph_weights_stakes(
                    netuid=netuid,
                    start_date=attempt_start_date,
                    end_date=use_end_date,
                    start_block=start_block,
                    end_block=use_end_block,
                    num_epochs=num_epochs,
                )
                break
            except Exception as e:
                msg = str(e).lower()
                # Normalize message and retry on any variant of epoch-start wording
                import re as _re
                msg_norm = _re.sub(r"[^a-z]+", " ", msg)
                if "epoch start" in msg_norm:
                    logger.warning(
                        f"No epoch-start blocks found from start={attempt_start_date}; retrying with additional backoff {bh}h"
                    )
                    last_err = e
                    continue
                last_err = e
                break
        if weights_stakes_data is None:
            assert last_err is not None
            # Fallback: try a broad date window without num_epochs so backend can choose epochs
            try:
                broad_start = (start_date - timedelta(days=7)) if start_date else None
                logger.warning(
                    f"Retrying with broad window start={broad_start}, end={original_end_date} (no num_epochs)"
                )
                weights_stakes_data = fetch_metagraph_weights_stakes(
                    netuid=netuid,
                    start_date=broad_start,
                    end_date=original_end_date,
                    start_block=start_block,
                    end_block=end_block,
                    num_epochs=None,
                )
                # Use the broad_start as the chosen window start for rewards alignment
                chosen_start_date = broad_start
            except Exception:
                raise last_err
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
            start_date=chosen_start_date,
            end_date=use_end_date,
            start_block=start_block,
            end_block=use_end_block,
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
