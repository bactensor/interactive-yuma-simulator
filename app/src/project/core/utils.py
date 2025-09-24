import logging
import re
from urllib.parse import urljoin
from datetime import datetime, timedelta

import requests
from typing import Optional, Dict, Any
import pandas as pd
from django.conf import settings
from django.core.cache import cache

UINT16_MAX = 65535.0
ONE_MILLION = 1_000_000.0

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class EpochStartNotFoundError(requests.HTTPError):
    """Raised when backend reports missing epoch-start blocks in the requested window."""


# TODO: refactor yuma-simulation package to accept hyperparameter values natively
def normalize(value: float, max_value: float) -> float:
    """Normalize a value to the [0,1] range based on a given maximum hyperparameter value."""
    try:
        return value / max_value
    except (TypeError, ZeroDivisionError):
        raise ValueError(f"Cannot normalize value={value} with max_value={max_value}")


_CACHE_KEY = "metagraph_client_session"


def _build_metagraph_query_params(
    *,
    netuid: int,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    start_block: Optional[int] = None,
    end_block: Optional[int] = None,
    num_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build query params for metagraph endpoints with clear validation.

    Rules:
    - Exactly one of start_date or start_block must be provided.
    - At least one of end_date, end_block, or num_epochs must be provided.
    - For end parameters, any combination is allowed (the server resolves precedence).
    """
    has_start_date = start_date is not None
    has_start_block = start_block is not None
    if has_start_date == has_start_block:
        raise ValueError("Provide exactly one of start_date or start_block")

    if end_date is None and end_block is None and num_epochs is None:
        raise ValueError("Provide at least one of end_date, end_block, or num_epochs")

    params: Dict[str, Any] = {"netuid": netuid}
    if start_block is not None:
        params["start_block"] = start_block
    else:
        params["start_date"] = start_date.isoformat()  # type: ignore[arg-type]

    if end_block is not None:
        params["end_block"] = end_block
    if end_date is not None:
        params["end_date"] = end_date.isoformat()
    # Only include num_epochs when no explicit end is provided.
    # Some backends reject requests that include both an end (date/block)
    # and num_epochs simultaneously.
    if num_epochs is not None and end_date is None and end_block is None:
        params["num_epochs"] = num_epochs

    return params


def get_metagraph_session() -> requests.Session:
    """
    Return a logged‐in Session for the external Django service.
    We cache it in Django’s cache so we only re-login once per hour.
    """
    sess = cache.get(_CACHE_KEY)
    if sess:
        return sess

    sess = requests.Session()
    login_url = urljoin(settings.MGRAPH_BASE_URL, "admin/login/")

    r1 = sess.get(login_url)
    m = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', r1.text)
    if not m:
        raise RuntimeError("Could not get CSRF token")
    token = m.group(1)

    resp = sess.post(
        login_url,
        data={
            "csrfmiddlewaretoken": token,
            "username": settings.MGRAPH_USERNAME,
            "password": settings.MGRAPH_PASSWORD,
            "next": "/admin/",
        },
        headers={"Referer": login_url},
        timeout=10,
    )
    resp.raise_for_status()
    if "admin/" not in resp.url:
        raise RuntimeError("Login failed")

    # cache for an hour (or however long your external session lives)
    cache.set(_CACHE_KEY, sess, 60 * 60)
    return sess



def fetch_metagraph_weights_stakes(
    *,
    netuid: int,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    start_block: Optional[int] = None,
    end_block: Optional[int] = None,
    num_epochs: Optional[int] = None,
) -> dict:
    sess = get_metagraph_session()
    url = urljoin(settings.MGRAPH_BASE_URL, "metagraph/weights_stakes/")
    params = _build_metagraph_query_params(
        netuid=netuid,
        start_date=start_date,
        end_date=end_date,
        start_block=start_block,
        end_block=end_block,
        num_epochs=num_epochs,
    )

    logger.debug("→ GET %s %r", url, params)
    r = sess.get(url, params=params, timeout=360)

    if not r.ok:
        headers = dict(r.headers)
        body = r.text
        try:
            parsed = r.json()
            err = parsed.get("error")
        except ValueError:
            parsed = None
            err = None

        logger.error(
            "metagraph weights/stakes fetch failed: %s %s (url=%s)\n"
            "Params: %r\n"
            "Response headers:\n%s\n"
            "Response body (first 500 chars):\n%s\n"
            "Parsed error: %r",
            r.status_code,
            r.reason,
            url,
            params,
            headers,
            body[:500],
            err,
        )

        msg = (
            f"HTTP {r.status_code} {r.reason} for {url} with params={params}. "
            f"Details: {err or (body[:200] if body else 'no body')}"
        )

        if err:
            err_lower = str(err).lower()
            if "epoch" in err_lower and "start" in err_lower:
                raise EpochStartNotFoundError(msg, response=r)

        raise requests.HTTPError(msg, response=r)

    return r.json()

def fetch_metagraph_data_with_initial_bonds(
    *,
    netuid: int,
    start_date: datetime,
    end_date: datetime,
) -> dict:
    """
    Fetch metagraph weights/stakes data and pad the range by one extra epoch
    to obtain bonds for simulator initialization.

    This function:
    1. Adjusts the requested window to include one extra epoch at the start
    2. Fetches weights/stakes for the expanded range
    3. Fetches rewards (bonds) for the first block to initialize the simulator
    4. Merges the data and returns it
    """
    #TODO Lack of bonds should be displayed to the user in the UI probably, not just logged - or epoch padding could be used as a fallback

    adjusted_start_date = start_date - timedelta(seconds=360 * 12)  # 360 blocks * 12 seconds

    weights_stakes_data = fetch_metagraph_weights_stakes(
        netuid=netuid,
        start_date=adjusted_start_date,
        end_date=end_date,
    )

    blocks = weights_stakes_data.get("blocks", [])

    # We only need the initial bonds to seed the simulator
    try:
        if blocks:
            first_block = blocks[0]
            logger.info("Fetching initial bonds for block %s, netuid %s", first_block, netuid)

            rewards_data = fetch_metagraph_rewards(
                netuid=netuid,
                start_block=first_block,
                end_block=first_block,
            )

            bonds_data = rewards_data.get("bonds", {})
            first_block_key = str(first_block)
            first_block_bonds = bonds_data.get(first_block_key, {})

            if first_block_bonds:
                weights_stakes_data["bonds"] = {first_block_key: first_block_bonds}
                logger.info("Initial bonds fetched successfully for block %s", first_block)
            else:
                logger.warning("No bonds data found for block %s. Simulator will start with zero bonds.", first_block)
                weights_stakes_data["bonds"] = {}
        else:
            logger.warning("No blocks available to fetch initial bonds")
            weights_stakes_data["bonds"] = {}
    except Exception as e:
        logger.warning("Failed to fetch initial bonds: %s (type=%s). Simulator will start with zero bonds.", e, type(e).__name__)
        weights_stakes_data["bonds"] = {}

    return weights_stakes_data


def fetch_metagraph_rewards(
    *,
    netuid: int,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    start_block: Optional[int] = None,
    end_block: Optional[int] = None,
    num_epochs: Optional[int] = None,
) -> dict:
    sess = get_metagraph_session()
    url = urljoin(settings.MGRAPH_BASE_URL, "metagraph/rewards/")
    params = _build_metagraph_query_params(
        netuid=netuid,
        start_date=start_date,
        end_date=end_date,
        start_block=start_block,
        end_block=end_block,
        num_epochs=num_epochs,
    )

    logger.debug("→ GET %s %r", url, params)
    r = sess.get(url, params=params, timeout=360)

    if not r.ok:
        headers = dict(r.headers)
        body = r.text
        try:
            parsed = r.json()
            err = parsed.get("error")
        except ValueError:
            parsed = None
            err = None

        logger.error(
            "metagraph rewards fetch failed: %s %s (url=%s)\n"
            "Params: %r\n"
            "Response headers:\n%s\n"
            "Response body (first 500 chars):\n%s\n"
            "Parsed error: %r",
            r.status_code,
            r.reason,
            url,
            params,
            headers,
            body[:500],
            err,
        )

        msg = (
            f"HTTP {r.status_code} {r.reason} for {url} with params={params}. "
            f"Details: {err or (body[:200] if body else 'no body')}"
        )
        raise requests.HTTPError(msg, response=r)

    return r.json()
