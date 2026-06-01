"""Discover candidate releases via SQLite (v12 SQL expansion)."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from discogs_recommender.config import AppConfig
from discogs_recommender.profile.owned import (
    filter_owned_releases,
    load_owned_sets,
    owned_release_ids,
)
from discogs_recommender.profile.sync import load_profile_releases

logger = logging.getLogger(__name__)


def _safe_int(value: object) -> int | None:
    """Parse DB/API ids; Discogs SQLite dumps may use empty strings for nulls."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _load_profile(config: AppConfig) -> list[dict[str, Any]]:
    path = config.paths.profile_releases
    if not path.is_file():
        raise FileNotFoundError(
            f"Profile not found at {path}. Run: python main.py sync_profile"
        )
    return load_profile_releases(path)


def _build_base_scores(
    profile: list[dict[str, Any]], config: AppConfig
) -> tuple[dict[int, float], dict[int, float], set[int]]:
    artist_scores: dict[int, float] = defaultdict(float)
    label_scores: dict[int, float] = defaultdict(float)
    owned = owned_release_ids(profile)

    for row in profile:
        weight = (
            config.profile.collection_weight
            if row["source"] == "collection"
            else config.profile.wantlist_weight
        )
        for aid in row.get("artist_ids") or []:
            artist_scores[int(aid)] += weight
        for lid in row.get("label_ids") or []:
            label_scores[int(lid)] += weight

    return dict(artist_scores), dict(label_scores), owned


def _expand_label_families(
    conn: sqlite3.Connection,
    label_scores: dict[int, float],
    config: AppConfig,
) -> None:
    dc = config.discover
    seed_ids = list(label_scores.keys())
    if not seed_ids:
        return

    placeholders = ",".join("?" * len(seed_ids))

    sublabels = pd.read_sql(
        f"""
        SELECT id, parent_id FROM label
        WHERE parent_id IN ({placeholders})
        """,
        conn,
        params=seed_ids,
    )
    for _, row in sublabels.iterrows():
        lid = _safe_int(row["id"])
        parent_id = _safe_int(row["parent_id"])
        if lid is None or parent_id is None:
            continue
        parent_score = label_scores.get(parent_id, 0)
        label_scores[lid] = label_scores.get(lid, 0) + parent_score * dc.family_decay

    parents = pd.read_sql(
        f"""
        SELECT id, parent_id FROM label
        WHERE id IN ({placeholders})
          AND parent_id IS NOT NULL AND CAST(parent_id AS TEXT) != ''
        """,
        conn,
        params=seed_ids,
    )
    new_parent_ids: set[int] = set()
    for _, row in parents.iterrows():
        pid = _safe_int(row["parent_id"])
        lid = _safe_int(row["id"])
        if pid is None or lid is None:
            continue
        child_score = label_scores.get(lid, 0)
        label_scores[pid] = label_scores.get(pid, 0) + child_score * dc.family_decay
        new_parent_ids.add(pid)

    if new_parent_ids:
        pp = ",".join("?" * len(new_parent_ids))
        siblings = pd.read_sql(
            f"SELECT id, parent_id FROM label WHERE parent_id IN ({pp})",
            conn,
            params=list(new_parent_ids),
        )
        for _, row in siblings.iterrows():
            lid = _safe_int(row["id"])
            parent_id = _safe_int(row["parent_id"])
            if lid is None or parent_id is None:
                continue
            if lid not in label_scores:
                parent_score = label_scores.get(parent_id, 0)
                label_scores[lid] = parent_score * dc.family_decay * dc.sibling_decay_factor

    logger.info("Labels after family expansion: %s", len(label_scores))


def _expand_cross_affinity(
    conn: sqlite3.Connection,
    artist_scores: dict[int, float],
    label_scores: dict[int, float],
    config: AppConfig,
) -> None:
    dc = config.discover
    genre = dc.genre

    top_artists = [
        aid
        for aid, _ in sorted(artist_scores.items(), key=lambda x: x[1], reverse=True)[
            : dc.top_artists_for_expansion
        ]
    ]
    if top_artists:
        ap = ",".join("?" * len(top_artists))
        artist_labels = pd.read_sql(
            f"""
            SELECT DISTINCT rl.label_id
            FROM release_label rl
            JOIN release_artist ra ON rl.release_id = ra.release_id
            JOIN release_genre rg ON rl.release_id = rg.release_id
            WHERE ra.artist_id IN ({ap}) AND ra.extra = 0 AND rg.genre = ?
            """,
            conn,
            params=[*top_artists, genre],
        )
        for _, row in artist_labels.iterrows():
            lid = _safe_int(row["label_id"])
            if lid is not None and lid not in label_scores:
                label_scores[lid] = dc.artist_to_label_decay

    top_labels = [
        lid
        for lid, _ in sorted(label_scores.items(), key=lambda x: x[1], reverse=True)[
            : dc.top_labels_for_expansion
        ]
    ]
    if top_labels:
        lp = ",".join("?" * len(top_labels))
        label_artists = pd.read_sql(
            f"""
            SELECT DISTINCT ra.artist_id
            FROM release_artist ra
            JOIN release_label rl ON ra.release_id = rl.release_id
            JOIN release_genre rg ON ra.release_id = rg.release_id
            WHERE rl.label_id IN ({lp}) AND ra.extra = 0 AND rg.genre = ?
            """,
            conn,
            params=[*top_labels, genre],
        )
        for _, row in label_artists.iterrows():
            aid = _safe_int(row["artist_id"])
            if aid is not None and aid not in artist_scores:
                artist_scores[aid] = dc.label_to_artist_decay

    logger.info(
        "After cross-expansion — artists: %s | labels: %s",
        len(artist_scores),
        len(label_scores),
    )


def _load_temp_tables(
    conn: sqlite3.Connection,
    artist_scores: dict[int, float],
    label_scores: dict[int, float],
    owned_releases: set[int],
    owned_masters: set[int],
) -> None:
    """Write affinity scores and owned sets into SQLite temp tables."""
    for tbl in (
        "tmp_artist_affinity",
        "tmp_label_affinity",
        "tmp_owned_release",
        "tmp_owned_master",
    ):
        conn.execute(f"DROP TABLE IF EXISTS {tbl}")

    conn.execute(
        "CREATE TEMP TABLE tmp_artist_affinity (artist_id INTEGER PRIMARY KEY, score REAL)"
    )
    conn.execute(
        "CREATE TEMP TABLE tmp_label_affinity (label_id INTEGER PRIMARY KEY, score REAL)"
    )
    conn.execute("CREATE TEMP TABLE tmp_owned_release (release_id INTEGER PRIMARY KEY)")
    conn.execute("CREATE TEMP TABLE tmp_owned_master (master_id INTEGER PRIMARY KEY)")

    conn.executemany("INSERT INTO tmp_artist_affinity VALUES (?, ?)", artist_scores.items())
    conn.executemany("INSERT INTO tmp_label_affinity VALUES (?, ?)", label_scores.items())
    conn.executemany(
        "INSERT INTO tmp_owned_release VALUES (?)", ((r,) for r in owned_releases)
    )
    conn.executemany(
        "INSERT INTO tmp_owned_master VALUES (?)", ((m,) for m in owned_masters)
    )
    conn.commit()
    logger.info(
        "Temp tables loaded — artists: %s | labels: %s | owned releases: %s | owned masters: %s",
        len(artist_scores),
        len(label_scores),
        len(owned_releases),
        len(owned_masters),
    )


def _build_style_temp_table(
    conn: sqlite3.Connection,
    profile: list[dict[str, Any]],
    config: AppConfig,
) -> None:
    """Build tmp_style_affinity from profile release styles, weighted by source."""
    pc = config.profile
    weight_map = {
        int(row["release_id"]): (
            pc.collection_weight if row.get("source") == "collection" else pc.wantlist_weight
        )
        for row in profile
    }
    release_ids = list(weight_map)
    style_raw: dict[str, float] = defaultdict(float)
    batch_size = config.discover.sql_batch_size

    for i in range(0, len(release_ids), batch_size):
        batch = release_ids[i : i + batch_size]
        p = ",".join("?" * len(batch))
        for rid, style in conn.execute(
            f"SELECT release_id, style FROM release_style WHERE release_id IN ({p})", batch
        ).fetchall():
            w = weight_map.get(int(rid), 1.0)
            if style:
                style_raw[str(style).strip()] += w

    if not style_raw:
        logger.warning("No styles found in profile — discovery bucket will be empty")
        return

    max_v = max(style_raw.values())
    style_aff = {k: v / max_v for k, v in style_raw.items()}

    conn.execute("DROP TABLE IF EXISTS tmp_style_affinity")
    conn.execute("CREATE TEMP TABLE tmp_style_affinity (style TEXT PRIMARY KEY, score REAL)")
    conn.executemany("INSERT INTO tmp_style_affinity VALUES (?, ?)", style_aff.items())
    conn.commit()
    logger.info("Style temp table: %s styles from profile", len(style_aff))


_CANDIDATE_SQL = """
    SELECT
        r.id      AS release_id,
        r.title,
        r.country,
        r.released,
        r.master_id,
        COALESCE(aa.max_score, 0.0) + COALESCE(la.max_score, 0.0) AS metadata_score
    FROM release r
    JOIN release_genre rg ON r.id = rg.release_id
    LEFT JOIN (
        SELECT ra.release_id, MAX(af.score) AS max_score
        FROM release_artist ra
        JOIN tmp_artist_affinity af ON ra.artist_id = af.artist_id
        WHERE ra.extra = 0
        GROUP BY ra.release_id
    ) aa ON r.id = aa.release_id
    LEFT JOIN (
        SELECT rl.release_id, MAX(lf.score) AS max_score
        FROM release_label rl
        JOIN tmp_label_affinity lf ON rl.label_id = lf.label_id
        GROUP BY rl.release_id
    ) la ON r.id = la.release_id
    WHERE rg.genre = ?
      AND r.id NOT IN (SELECT release_id FROM tmp_owned_release)
      AND (
          r.master_id IS NULL
          OR r.master_id = 0
          OR r.master_id NOT IN (SELECT master_id FROM tmp_owned_master)
      )
      AND {compilation_clause}
    ORDER BY metadata_score DESC
    LIMIT ?
"""

_IS_COMPILATION = (
    "r.id IN (SELECT release_id FROM release_artist"
    " WHERE LOWER(TRIM(artist_name)) = 'various')"
)
_NOT_COMPILATION = (
    "r.id NOT IN (SELECT release_id FROM release_artist"
    " WHERE LOWER(TRIM(artist_name)) = 'various')"
)

_DISCOVERY_SQL = """
    SELECT
        r.id      AS release_id,
        r.title,
        r.country,
        r.released,
        r.master_id,
        0.0       AS metadata_score
    FROM release r
    JOIN release_genre rg ON r.id = rg.release_id
    JOIN (
        SELECT rs.release_id, MAX(af.score) AS max_style
        FROM release_style rs
        JOIN tmp_style_affinity af ON rs.style = af.style
        GROUP BY rs.release_id
    ) sa ON r.id = sa.release_id
    WHERE rg.genre = ?
      AND r.id NOT IN (SELECT release_id FROM tmp_owned_release)
      AND (
          r.master_id IS NULL
          OR r.master_id = 0
          OR r.master_id NOT IN (SELECT master_id FROM tmp_owned_master)
      )
      AND {compilation_clause}
    ORDER BY sa.max_style DESC
    LIMIT ?
"""


def _query_all_electronic(
    conn: sqlite3.Connection,
    config: AppConfig,
) -> pd.DataFrame:
    """Build the candidate pool from three buckets:

    - Affinity (45% default): releases ranked by label/artist affinity score.
    - Discovery (45% default): releases ranked purely by style affinity, no
      relationship to the collection required — surfaces unknown artists/labels.
    - Compilations (10% default): VA releases ranked by affinity score.

    All three are merged and deduplicated on release_id before return.
    """
    dc = config.discover
    total = dc.candidate_pool_limit
    comp_limit = max(1, round(total * dc.compilation_pool_fraction))
    disc_limit = max(1, round(total * dc.discovery_pool_fraction))
    affinity_limit = total - comp_limit - disc_limit

    affinity = pd.read_sql(
        _CANDIDATE_SQL.format(compilation_clause=_NOT_COMPILATION),
        conn,
        params=[dc.genre, affinity_limit],
    )
    comp = pd.read_sql(
        _CANDIDATE_SQL.format(compilation_clause=_IS_COMPILATION),
        conn,
        params=[dc.genre, comp_limit],
    )
    discovery = pd.read_sql(
        _DISCOVERY_SQL.format(compilation_clause=_NOT_COMPILATION),
        conn,
        params=[dc.genre, disc_limit],
    )

    df = (
        pd.concat([affinity, comp, discovery], ignore_index=True)
        .drop_duplicates(subset=["release_id"])
        .reset_index(drop=True)
    )
    logger.info(
        "Raw candidate pool: %s releases (%s affinity, %s style-discovery, %s compilation)",
        len(df),
        len(affinity),
        len(discovery),
        len(comp),
    )
    return df


def _batched_query(
    conn: sqlite3.Connection,
    query_template: str,
    ids: list[int],
    batch_size: int,
) -> pd.DataFrame:
    if not ids:
        return pd.DataFrame()
    frames = []
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        placeholders = ",".join("?" * len(batch))
        sql = query_template.format(placeholders=placeholders)
        frames.append(pd.read_sql(sql, conn, params=batch))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _enrich_candidates(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    config: AppConfig,
) -> pd.DataFrame:
    """Attach display columns (styles, artist/label names and IDs) to the candidate pool.

    metadata_score is already set by the SQL query and is preserved as-is.
    """
    if df.empty:
        return df

    batch = config.discover.sql_batch_size
    ids = df["release_id"].astype(int).tolist()

    styles_df = _batched_query(
        conn,
        "SELECT release_id, style FROM release_style WHERE release_id IN ({placeholders})",
        ids,
        batch,
    )
    artists_df = _batched_query(
        conn,
        "SELECT release_id, artist_id, artist_name FROM release_artist "
        "WHERE release_id IN ({placeholders}) AND extra = 0",
        ids,
        batch,
    )
    labels_df = _batched_query(
        conn,
        "SELECT release_id, label_id, label_name FROM release_label "
        "WHERE release_id IN ({placeholders})",
        ids,
        batch,
    )

    styles_map = styles_df.groupby("release_id")["style"].apply(list).to_dict()
    artist_names_map = artists_df.groupby("release_id")["artist_name"].apply(list).to_dict()
    artist_ids_map = artists_df.groupby("release_id")["artist_id"].apply(list).to_dict()
    label_names_map = labels_df.groupby("release_id")["label_name"].apply(list).to_dict()
    label_ids_map = labels_df.groupby("release_id")["label_id"].apply(list).to_dict()

    df = df.copy()
    df["styles"] = df["release_id"].map(lambda x: styles_map.get(x, []))
    df["artists"] = df["release_id"].map(lambda x: artist_names_map.get(x, []))
    df["artist_ids"] = df["release_id"].map(lambda x: artist_ids_map.get(x, []))
    df["labels"] = df["release_id"].map(lambda x: label_names_map.get(x, []))
    df["label_ids"] = df["release_id"].map(lambda x: label_ids_map.get(x, []))

    return df.sort_values("metadata_score", ascending=False).reset_index(drop=True)


def discover_candidates(config: AppConfig, *, force: bool = False) -> Path:
    """Build candidate pool and write ``exports/candidates.csv``."""
    out_path = config.paths.candidates_file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.is_file() and not force:
        logger.info("Candidates cache at %s (use --force to rebuild)", out_path)
        return out_path

    if not config.paths.db_path.is_file():
        raise FileNotFoundError(f"SQLite DB not found: {config.paths.db_path}")

    profile = _load_profile(config)
    artist_scores, label_scores, owned_releases = _build_base_scores(profile, config)
    logger.info(
        "Profile: %s releases | %s artists | %s labels",
        len(owned_releases),
        len(artist_scores),
        len(label_scores),
    )

    conn = sqlite3.connect(config.paths.db_path)
    try:
        owned_release_set, owned_master_set = load_owned_sets(profile, conn)
        logger.info(
            "Excluding %s owned releases and %s owned masters",
            len(owned_release_set),
            len(owned_master_set),
        )
        _expand_label_families(conn, label_scores, config)
        _expand_cross_affinity(conn, artist_scores, label_scores, config)
        _load_temp_tables(conn, artist_scores, label_scores, owned_release_set, owned_master_set)
        _build_style_temp_table(conn, profile, config)
        df = _query_all_electronic(conn, config)
        before_owned = len(df)
        df = filter_owned_releases(df, owned_release_set, owned_master_set, conn=conn)
        if before_owned != len(df):
            logger.info(
                "Post-query owned filter: %s → %s releases", before_owned, len(df)
            )
        df = _enrich_candidates(conn, df, config)
    finally:
        conn.close()

    # Serialize list columns as JSON strings for CSV
    export = df.copy()
    for col in ("styles", "artists", "artist_ids", "labels", "label_ids"):
        if col in export.columns:
            export[col] = export[col].apply(
                lambda v: json.dumps(v if isinstance(v, list) else list(v or []))
            )

    export.to_csv(out_path, index=False)
    logger.info("Wrote %s candidates to %s", len(export), out_path)
    return out_path
