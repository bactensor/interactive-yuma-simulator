
import logging
import os
import time
import numpy as np
import torch
import bittensor as bt
from multiprocessing import Pool
from .experiment_setup import ExperimentSetup
from typing import Optional
from typing import Dict, List, Set, Tuple

HotkeyTuple = Tuple[str, bool, bool]

logger = logging.getLogger(__name__)

_archive_session = None

def get_archive_session():
    global _archive_session
    if _archive_session is None:
        _archive_session = bt.subtensor("archive")
    return _archive_session

def ensure_tensor_on_cpu(obj):
    return obj.cpu() if isinstance(obj, torch.Tensor) else obj


def check_file_corruption(file_path):
    """
    True if torch.load or torch.jit.load can read the file, else False.
    """
    try:
        torch.load(file_path, map_location="cpu")
        logger.debug(f"{file_path} loaded with torch.load")
        return True
    except Exception as e:
        logger.warning(f"{file_path} failed torch.load: {e}")
        try:
            torch.jit.load(file_path, map_location="cpu")
            logger.debug(f"{file_path} loaded with torch.jit.load")
            return True
        except Exception as e:
            logger.error(f"{file_path} failed torch.jit.load: {e}")
            return False


def download_metagraph(netuid, start_block, file_prefix, max_retries=5, retry_delay=5):
    """
    1) If file for block already exists and isn't corrupted, we skip.
    2) Otherwise, we download a new one, save, verify; if corrupted, remove and go to next block.
    """
    block = start_block
    for attempt in range(max_retries):
        file_path = f"{file_prefix}_{block}.pt"

        # 1) If file exists and is valid, skip download
        if os.path.isfile(file_path):
            if check_file_corruption(file_path):
                logger.info(f"Block {block}: File already valid, skipping.")
                return block
            else:
                logger.warning(f"Block {block}: Corrupted file found, removing.")
                os.remove(file_path)

        # 2) Download a fresh file
        try:
            logger.info(f"Block {block}: Downloading metagraph --> {file_path}")
            archive_subtensor = bt.subtensor("archive")
            meta = archive_subtensor.metagraph(netuid=netuid, block=block, lite=False)
            meta_dict = {
                "netuid": meta.netuid,
                "block": meta.block,
                "S": ensure_tensor_on_cpu(meta.S),
                "W": ensure_tensor_on_cpu(meta.W),
                "hotkeys": meta.hotkeys,
            }
            torch.save(meta_dict, file_path)

            # 3) Check corruption
            if check_file_corruption(file_path):
                logger.debug(f"Block {block}: File verified successfully.")
                return block
            else:
                logger.error(f"Block {block}: File is corrupted, removing.")
                os.remove(file_path)

        except Exception as e:
            logger.error(f"Block {block}: Error: {e}")

        # Move on to the next block if this one fails
        block += 1
        time.sleep(retry_delay)

    # If we reach here, max_retries are used up
    raise RuntimeError(
        f"Failed to download metagraph for netuid={netuid} "
        f"starting at block={start_block} after {max_retries} retries."
    )



def load_metas_from_directory(storage_path: str, epochs_num: int) -> list[torch.Tensor]:
    metas = []

    logger.info(f"Checking directory: {storage_path}")
    if not os.path.exists(storage_path):
        logger.error(f"Directory does not exist: {storage_path}")
        return metas

    files = os.listdir(storage_path)
    logger.debug(f"Found files: {files}")

    num_files = len(files)

    if epochs_num > num_files:
        logger.warning(f"Requested {epochs_num} epochs, but only {num_files} files available.")
        epochs_num = num_files

    files = files[:epochs_num]

    for filename in sorted(files):
        file_path = os.path.join(storage_path, filename)
        logger.debug(f"Processing file: {file_path}")

        try:
            data = torch.load(file_path, map_location="cpu")
            metas.append(data)
            logger.debug(f"Loaded file: {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")

    logger.info(f"Loaded {len(metas)} metagraphs (Requested: {epochs_num}).")
    return metas


class DownloadMetagraph:
    def __init__(self, setup=None):
        self.setup = setup if setup else ExperimentSetup()

    def run(self):
        os.makedirs(self.setup.metagraph_storage_path, exist_ok=True)
        logger.info(f"Created directory: {self.setup.metagraph_storage_path}")

        # Prepare the (netuid, block, prefix) args
        args = []
        for netuid in self.setup.netuids:
            for conceal_period in self.setup.conceal_periods:
                for dp in range(self.setup.data_points):
                    block_early = self.setup.start_block + self.setup.tempo * dp
                    prefix_early = os.path.join(
                        self.setup.metagraph_storage_path,
                        f"netuid{netuid}_block{block_early}",
                    )
                    args.append((netuid, block_early, prefix_early))

                    block_late = self.setup.start_block + self.setup.tempo * (
                        dp + conceal_period
                    )
                    prefix_late = os.path.join(
                        self.setup.metagraph_storage_path,
                        f"netuid{netuid}_block{block_late}",
                    )
                    args.append((netuid, block_late, prefix_late))

        args = list(set(args))  # remove duplicates
        logger.debug(f"Prepared download arguments: {args}")

        try:
            with Pool(processes=self.setup.processes) as pool:
                pool.starmap(download_metagraph, args)
            logger.info("DownloadMetagraph run completed.")
        except Exception as e:
            logger.error("Error occurred during metagraph download in pool.", e)
            raise

def slot_count(hotkeys_by_blk: Dict[int, List[str]]) -> int:
    """Return 256 or 1024 depending on subnet size."""
    return len(next(iter(hotkeys_by_blk.values())))

def build_S_tensor(stakes_map: Dict[str, float], n_slots: int) -> torch.Tensor:
    S = torch.zeros(n_slots, dtype=torch.float32)
    for uid, stake in stakes_map.items():
        S[int(uid)] = float(stake)
    return S

def build_W_tensor(weight_map: Dict[str, Dict[str, float]],
                   n_slots: int) -> torch.Tensor:
    W = torch.zeros((n_slots, n_slots), dtype=torch.float32)
    for src_uid, row in weight_map.items():
        i = int(src_uid)
        for tgt_uid, w in row.items():
            j = int(tgt_uid)
            W[i, j] = float(w)
    return W

def pick_validators(
    hotkeys_by_blk: Dict[int, List[HotkeyTuple]],
    min_stake: float = 1000.0,
    stakes: Dict[str, Dict[str, float]] | None = None,
) -> List[str]:
    """
    Return hotkey strings that ever had is_validator == True
    (optionally stake ≥ min_stake if stakes is provided).
    """
    vals: Set[str] = set()

    for blk_int, slot_list in hotkeys_by_blk.items():
        for uid, slot in enumerate(slot_list):
            hk, is_val, is_active = slot
            if not (hk and is_val and is_active):
                continue

            if stakes is not None:
                stake_amt = stakes[str(blk_int)].get(str(uid), 0.0)
                if stake_amt < min_stake:
                    continue

            vals.add(hk)

    return sorted(vals)

def run_block_diagnostics(block: int,
                          netuid: int,
                          S: torch.Tensor,
                          W: torch.Tensor,
                          hotkeys: List[str],
                          tol: float = 1e-6) -> None:
    """
    Compare local S, W, hotkeys against on‑chain metagraph for a single block.
    Logs summary lines; raises nothing.
    """
    try:
        st = get_archive_session()
        meta = st.metagraph(netuid=netuid,
                            block=block,
                            lite=False)
    except Exception as e:
        logger.warning("Diag fetch failed for %d: %s", block, e)
        return

    meta_S = torch.from_numpy(meta.S)
    meta_W = torch.from_numpy(meta.W)

    # stakes
    miss_s = torch.nonzero((meta_S - S).abs() > tol, as_tuple=False)
    if miss_s.numel():
        logger.error("STAKE diff @%d (%d uids)", block, miss_s.numel())

    # weights
    diff_W = (meta_W - W).abs()
    mask   = ~((meta_W == 1.0) & (W == 0.0))
    miss_w = torch.nonzero(diff_W.gt(tol) & mask, as_tuple=False)
    if miss_w.numel():
        logger.error("WEIGHT diff @%d (%d cells)", block, miss_w.numel())

    # hotkeys
    miss_h = [i for i, (o, l) in enumerate(zip(meta.hotkeys, hotkeys)) if o != l]
    if miss_h:
        logger.error("HOTKEY diff @%d (%d slots)", block, len(miss_h))


if __name__ == "__main__":
    DownloadMetagraph().run()
