"""Discogs REST API client with rate limiting and retries."""

from __future__ import annotations

import logging
import random
import time
from json import JSONDecodeError
from typing import Any

import requests

from discogs_recommender.config import DiscogsApiConfig

logger = logging.getLogger(__name__)


class DiscogsClient:
    def __init__(self, config: DiscogsApiConfig) -> None:
        if not config.user_token or not config.username:
            raise ValueError(
                "DISCOGS_USER_TOKEN and DISCOGS_USERNAME must be set in .env"
            )
        self.config = config
        self._sleep_s = 60.0 / max(config.requests_per_min, 1)
        self._ua = (
            f"DiscogsRecommender/0.1 (+https://www.discogs.com/user/{config.username})"
        )

    def _headers(self) -> dict[str, str]:
        return {
            "User-Agent": self._ua,
            "Authorization": f"Discogs token={self.config.user_token}",
            "Accept": "application/json",
            "Connection": "keep-alive",
        }

    def _backoff(self, attempt: int) -> float:
        cap = 120.0
        base = self._sleep_s
        return min(cap, base * (1.9 ** (attempt - 1))) + random.uniform(0, 0.9)

    def get(self, url: str, params: dict[str, Any] | None = None, *, what: str = "api") -> dict:
        params = params or {}
        for attempt in range(1, self.config.max_retries + 1):
            try:
                time.sleep(self._sleep_s)
                response = requests.get(
                    url,
                    headers=self._headers(),
                    params=params,
                    timeout=self.config.timeout_s,
                )
                if response.status_code in (429, 500, 502, 503, 504):
                    sleep_for = self._backoff(attempt)
                    logger.warning(
                        "[%s retry %s] HTTP %s; sleep %.1fs",
                        what,
                        attempt,
                        response.status_code,
                        sleep_for,
                    )
                    time.sleep(sleep_for)
                    continue
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, JSONDecodeError) as exc:
                sleep_for = self._backoff(attempt)
                logger.warning(
                    "[%s retry %s] %s; sleep %.1fs",
                    what,
                    attempt,
                    type(exc).__name__,
                    sleep_for,
                )
                time.sleep(sleep_for)
        raise RuntimeError(f"Failed to fetch {what} after {self.config.max_retries} retries")

    def paginate(
        self,
        url: str,
        result_key: str,
        *,
        what: str,
        per_page: int | None = None,
    ) -> list[dict]:
        per_page = per_page or self.config.per_page
        page = 1
        rows: list[dict] = []
        while True:
            data = self.get(url, {"page": page, "per_page": per_page}, what=what)
            rows.extend(data.get(result_key, []))
            pagination = data.get("pagination") or {}
            if page >= pagination.get("pages", 1):
                break
            page += 1
        return rows
