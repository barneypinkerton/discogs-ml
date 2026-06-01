"""Fetch collection and wantlist from Discogs API."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from discogs_recommender.config import AppConfig
from discogs_recommender.profile.discogs_client import DiscogsClient

logger = logging.getLogger(__name__)


def _release_row(basic_info: dict, *, source: str) -> dict[str, Any] | None:
    release_id = basic_info.get("id")
    if not release_id:
        return None
    artists_raw = basic_info.get("artists") or []
    labels_raw = basic_info.get("labels") or []
    return {
        "release_id": int(release_id),
        "title": basic_info.get("title"),
        "artist_ids": [
            int(a["id"])
            for a in artists_raw
            if isinstance(a, dict) and a.get("id") is not None
        ],
        "artist_names": [
            a.get("name", "") for a in artists_raw if isinstance(a, dict)
        ],
        "label_ids": [
            int(l["id"])
            for l in labels_raw
            if isinstance(l, dict) and l.get("id") is not None
        ],
        "label_names": [
            l.get("name", "") for l in labels_raw if isinstance(l, dict)
        ],
        "genres": basic_info.get("genres") or [],
        "styles": basic_info.get("styles") or [],
        "year": basic_info.get("year"),
        "source": source,
    }


def fetch_profile_releases(config: AppConfig) -> list[dict[str, Any]]:
    client = DiscogsClient(config.discogs)
    username = config.discogs.username
    rows: list[dict[str, Any]] = []

    collection_url = (
        f"https://api.discogs.com/users/{username}/collection/folders/0/releases"
    )
    logger.info("Fetching collection...")
    for item in client.paginate(collection_url, "releases", what="collection"):
        bi = item.get("basic_information") or {}
        row = _release_row(bi, source="collection")
        if row:
            rows.append(row)
    logger.info("Collection releases: %s", len(rows))

    wantlist_url = f"https://api.discogs.com/users/{username}/wants"
    wantlist_count = 0
    logger.info("Fetching wantlist...")
    for item in client.paginate(wantlist_url, "wants", what="wantlist"):
        bi = item.get("basic_information") or item
        row = _release_row(bi, source="wantlist")
        if row:
            rows.append(row)
            wantlist_count += 1
    logger.info("Wantlist releases: %s", wantlist_count)

    # Deduplicate: collection wins over wantlist for same release_id
    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        rid = row["release_id"]
        if rid not in by_id or row["source"] == "collection":
            by_id[rid] = row
    return list(by_id.values())


def sync_profile(config: AppConfig, *, force: bool = False) -> Path:
    """Download profile and write ``profile/releases.json``."""
    out_path = config.paths.profile_releases
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.is_file() and not force:
        logger.info("Profile cache present at %s (use --force to refresh)", out_path)
        return out_path

    releases = fetch_profile_releases(config)
    payload = {
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "username": config.discogs.username,
        "release_count": len(releases),
        "releases": releases,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %s releases to %s", len(releases), out_path)
    return out_path


def load_profile_releases(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("releases", []))
