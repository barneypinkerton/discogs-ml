"""Owned release and master sets for excluding already-collected music."""

from __future__ import annotations

import sqlite3
from typing import Any

import pandas as pd


def valid_master_id(value: object) -> int | None:
    """Return a positive Discogs master id, or None if missing/invalid."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        mid = int(value)
    except (ValueError, TypeError):
        return None
    return mid if mid > 0 else None


def owned_release_ids(profile: list[dict[str, Any]]) -> set[int]:
    return {int(row["release_id"]) for row in profile}


def owned_collection_artist_ids(profile: list[dict[str, Any]]) -> set[int]:
    """Discogs artist ids from collection releases (not wantlist)."""
    ids: set[int] = set()
    for row in profile:
        if row.get("source") != "collection":
            continue
        for aid in row.get("artist_ids") or []:
            ids.add(int(aid))
    return ids


def _parse_artist_ids(value: object) -> list[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        import json

        try:
            raw = json.loads(value)
        except json.JSONDecodeError:
            return []
    else:
        return []
    ids: list[int] = []
    for item in raw:
        try:
            ids.append(int(item))
        except (ValueError, TypeError):
            continue
    return ids


def filter_owned_artists(
    df: pd.DataFrame,
    owned_artists: set[int],
    *,
    artist_ids_column: str = "artist_ids",
) -> pd.DataFrame:
    """Drop candidates that credit any artist already in the user's collection."""
    if df.empty or not owned_artists:
        return df

    def credits_owned_artist(row: pd.Series) -> bool:
        return bool(set(_parse_artist_ids(row.get(artist_ids_column))) & owned_artists)

    mask = df.apply(credits_owned_artist, axis=1)
    return df[~mask].reset_index(drop=True)


def owned_master_ids(
    conn: sqlite3.Connection,
    release_ids: set[int],
    *,
    batch_size: int = 500,
) -> set[int]:
    """Master ids for any owned release (collection + wantlist)."""
    if not release_ids:
        return set()
    masters: set[int] = set()
    ids = list(release_ids)
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"SELECT master_id FROM release WHERE id IN ({placeholders})",
            batch,
        ).fetchall()
        for (mid,) in rows:
            parsed = valid_master_id(mid)
            if parsed is not None:
                masters.add(parsed)
    return masters


def load_owned_sets(
    profile: list[dict[str, Any]],
    conn: sqlite3.Connection,
) -> tuple[set[int], set[int]]:
    release_ids = owned_release_ids(profile)
    master_ids = owned_master_ids(conn, release_ids)
    return release_ids, master_ids


def release_master_ids(
    conn: sqlite3.Connection,
    release_ids: list[int],
    *,
    batch_size: int = 500,
) -> dict[int, int | None]:
    """Map release id → master id (or None)."""
    result: dict[int, int | None] = {}
    if not release_ids:
        return result
    for i in range(0, len(release_ids), batch_size):
        batch = release_ids[i : i + batch_size]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"SELECT id, master_id FROM release WHERE id IN ({placeholders})",
            batch,
        ).fetchall()
        for rid, mid in rows:
            result[int(rid)] = valid_master_id(mid)
    return result


def filter_owned_releases(
    df: pd.DataFrame,
    owned_releases: set[int],
    owned_masters: set[int],
    *,
    conn: sqlite3.Connection | None = None,
) -> pd.DataFrame:
    """Drop rows the user already has (exact release or another pressing)."""
    if df.empty or (not owned_releases and not owned_masters):
        return df

    out = df[~df["release_id"].isin(owned_releases)]
    if not owned_masters:
        return out.reset_index(drop=True)

    if "master_id" in out.columns:
        master_series = out["master_id"].map(valid_master_id)
    elif conn is not None:
        lookup = release_master_ids(
            conn, out["release_id"].astype(int).tolist()
        )
        master_series = out["release_id"].map(lookup)
    else:
        return out.reset_index(drop=True)

    duplicate_master = master_series.isin(owned_masters)
    return out[~duplicate_master].reset_index(drop=True)
