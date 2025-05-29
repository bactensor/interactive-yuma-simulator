import logging
import re
from urllib.parse import urljoin
from datetime import datetime

import requests
import pandas as pd
from django.conf import settings
from django.core.cache import cache

UINT16_MAX = 65535.0
ONE_MILLION = 1_000_000.0

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# TODO: refactor yuma-simulation package to accept hyperparameter values natively
def normalize(value: float, max_value: float) -> float:
    """Normalize a value to the [0,1] range based on a given maximum hyperparameter value."""
    try:
        return value / max_value
    except (TypeError, ZeroDivisionError):
        raise ValueError(f"Cannot normalize value={value} with max_value={max_value}")


_CACHE_KEY = "metagraph_client_session"


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


def fetch_metagraph_data(
    start_date: datetime,
    end_date: datetime,
    netuid: int,
) -> dict:
    sess = get_metagraph_session()
    url = urljoin(settings.MGRAPH_BASE_URL, "metagraph-data/")
    params = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "netuid": netuid,
    }

    logger.debug("→ GET %s %r", url, params)
    r = sess.get(url, params=params, timeout=360)

    if not r.ok:
        headers = dict(r.headers)
        body = r.text
        try:
            err = r.json().get("error")
        except ValueError:
            err = None

        logger.error(
            "metagraph data fetch failed: %s %s\n"
            "Response headers:\n%s\n"
            "Response body (first 500 chars):\n%s\n"
            "Parsed error: %r",
            r.status_code,
            r.reason,
            headers,
            body[:500],
            err,
        )

        http_err = requests.HTTPError(f"{r.status_code} {r.reason}", response=r)
        raise http_err

    return r.json()
