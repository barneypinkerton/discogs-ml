"""Filter, enrich, and rank candidates into final recommendations."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from discogs_recommender.config import AppConfig
from discogs_recommender.profile.discogs_client import DiscogsClient
from discogs_recommender.ui.preferences import UserPreferences
from discogs_recommender.profile.owned import (
    _parse_artist_ids,
    filter_owned_releases,
    load_owned_sets,
    release_master_ids,
    valid_master_id,
)
from discogs_recommender.profile.sync import load_profile_releases

logger = logging.getLogger(__name__)


def _parse_list_column(value: object) -> list:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return [value]
    return list(value)


def filter_compilations(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["artists"].apply(
            lambda artists: not any(
                str(name).strip().lower() == "various"
                for name in _parse_list_column(artists)
            )
        )
    ].reset_index(drop=True)


def _normalize_title(title: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(title or "").lower())


def _primary_artist_key(row: pd.Series) -> int | str | None:
    artist_ids = _parse_artist_ids(row.get("artist_ids"))
    if artist_ids:
        return artist_ids[0]
    artists = _parse_list_column(row.get("artists"))
    return artists[0] if artists else None


def _primary_label_key(row: pd.Series) -> int | str | None:
    label_ids = _parse_artist_ids(row.get("label_ids"))
    if label_ids:
        return label_ids[0]
    labels = _parse_list_column(row.get("labels"))
    return labels[0] if labels else None


def _work_dedup_key(row: pd.Series) -> tuple[int | str | None, str]:
    return (_primary_artist_key(row), _normalize_title(row.get("title")))


def ensure_master_ids(
    df: pd.DataFrame, conn: sqlite3.Connection | None
) -> pd.DataFrame:
    if "master_id" in df.columns or conn is None or df.empty:
        return df
    lookup = release_master_ids(conn, df["release_id"].astype(int).tolist())
    out = df.copy()
    out["master_id"] = out["release_id"].map(lookup)
    return out


def _build_alias_canonical_map(
    conn: sqlite3.Connection,
    artist_ids: set[int],
    *,
    batch_size: int = 500,
) -> dict[int, int]:
    """Map every artist_id to a canonical ID so aliases share one cap bucket.

    Discogs stores aliases by name only (alias_artist_id is always NULL), so we
    resolve name → id via the artist table, then find connected components and
    assign the minimum id in each cluster as the canonical id.
    """
    if not artist_ids:
        return {}

    ids = list(artist_ids)

    # 1. Collect alias names for all primary artists
    alias_rows: list[tuple[int, str]] = []
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        p = ",".join("?" * len(batch))
        alias_rows.extend(
            conn.execute(
                f"SELECT artist_id, alias_name FROM artist_alias WHERE artist_id IN ({p})",
                batch,
            ).fetchall()
        )

    if not alias_rows:
        return {aid: aid for aid in artist_ids}

    # 2. Resolve alias names → artist IDs via the artist table
    alias_names = list({name for _, name in alias_rows})
    name_to_ids: dict[str, list[int]] = defaultdict(list)
    for i in range(0, len(alias_names), batch_size):
        batch = alias_names[i : i + batch_size]
        p = ",".join("?" * len(batch))
        for aid, name in conn.execute(
            f"SELECT id, name FROM artist WHERE name IN ({p})", batch
        ).fetchall():
            name_to_ids[name].append(aid)

    # 3. Build undirected adjacency graph
    graph: dict[int, set[int]] = defaultdict(set)
    for artist_id, alias_name in alias_rows:
        for alias_id in name_to_ids.get(alias_name, []):
            if alias_id != artist_id:
                graph[artist_id].add(alias_id)
                graph[alias_id].add(artist_id)

    # 4. BFS to find connected components; canonical = min id in cluster
    canonical: dict[int, int] = {}
    visited: set[int] = set()
    all_ids = set(ids) | {aid for group in name_to_ids.values() for aid in group}

    for start in all_ids:
        if start in visited:
            continue
        component: set[int] = set()
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            queue.extend(graph[node] - visited)
        canon = min(component)
        for a in component:
            canonical[a] = canon

    # Artists with no aliases map to themselves
    for aid in artist_ids:
        if aid not in canonical:
            canonical[aid] = aid

    return canonical


def build_final_recommendations(
    df: pd.DataFrame,
    config: AppConfig,
    conn: sqlite3.Connection | None = None,
    alias_map: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Select up to top_n_export rows with artist/label caps and no duplicate works.

    alias_map collapses Discogs artist aliases into a single cap bucket so that
    e.g. Technasia and its aliases don't each get their own max_per_artist slots.
    """
    sc = config.score
    if df.empty:
        return df

    ranked = ensure_master_ids(df, conn)
    artist_counts: dict[int | str, int] = {}
    label_counts: dict[int | str, int] = {}
    seen_masters: set[int] = set()
    seen_works: set[tuple[int | str | None, str]] = set()
    keep_rows: list[pd.Series] = []

    for _, row in ranked.iterrows():
        if len(keep_rows) >= sc.top_n_export:
            break

        artist = _primary_artist_key(row)
        canonical_artist = (
            alias_map.get(artist, artist)
            if alias_map is not None and isinstance(artist, int)
            else artist
        )
        if canonical_artist is not None and artist_counts.get(canonical_artist, 0) >= sc.max_per_artist:
            continue

        label = _primary_label_key(row)
        if label is not None and label_counts.get(label, 0) >= sc.max_per_label:
            continue

        master = valid_master_id(row.get("master_id"))
        if master is not None:
            if master in seen_masters:
                continue
            seen_masters.add(master)
        else:
            work_key = _work_dedup_key(row)
            if work_key in seen_works:
                continue
            seen_works.add(work_key)

        if canonical_artist is not None:
            artist_counts[canonical_artist] = artist_counts.get(canonical_artist, 0) + 1
        if label is not None:
            label_counts[label] = label_counts.get(label, 0) + 1
        keep_rows.append(row)

    return pd.DataFrame(keep_rows).reset_index(drop=True)


def apply_diversity_cap(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Legacy helper; prefer :func:`build_final_recommendations`."""
    return build_final_recommendations(df, config, conn=None)


def enrich_have_want(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Fetch master-level community have/want counts for each row in *df*.

    For releases linked to a master, fetches the main release's have/want and
    uses those as the primary values. This means scoring reflects how well-known
    the work is across all pressings, not just the specific one recommended.
    The original pressing-level stats are preserved in ``release_have_count`` /
    ``release_want_count`` for reference.
    """
    if df.empty:
        return df

    limit = config.score.enrich_have_want_limit
    if limit <= 0:
        df = df.copy()
        df["have_count"] = np.nan
        df["want_count"] = np.nan
        df["release_have_count"] = np.nan
        df["release_want_count"] = np.nan
        return df

    if len(df) > limit:
        logger.warning(
            "enrich_have_want_limit=%s but %s rows to export; only first %s enriched",
            limit,
            len(df),
            limit,
        )

    client = DiscogsClient(config.discogs)
    df = df.copy()
    df["have_count"] = np.nan
    df["want_count"] = np.nan
    df["release_have_count"] = np.nan
    df["release_want_count"] = np.nan
    subset = df.head(limit)

    # Cache release payloads within this call to avoid re-fetching main releases
    _release_cache: dict[int, dict] = {}

    def _fetch_release(rid: int) -> dict:
        if rid not in _release_cache:
            _release_cache[rid] = client.get(
                f"https://api.discogs.com/releases/{rid}",
                what=f"release_{rid}",
            )
        return _release_cache[rid]

    for i, (idx, rid) in enumerate(subset["release_id"].items()):
        rid = int(rid)
        try:
            data = _fetch_release(rid)
            community = data.get("community") or {}
            release_have = community.get("have")
            release_want = community.get("want")
            df.at[idx, "release_have_count"] = release_have
            df.at[idx, "release_want_count"] = release_want

            # Default to release-level; upgrade to master's main release if available
            have, want = release_have, release_want
            master_id = valid_master_id(
                df.at[idx, "master_id"] if "master_id" in df.columns else None
            )
            if master_id:
                try:
                    master_data = client.get(
                        f"https://api.discogs.com/masters/{master_id}",
                        what=f"master_{master_id}",
                    )
                    main_rid = master_data.get("main_release")
                    if main_rid:
                        main_data = _fetch_release(int(main_rid))
                        main_community = main_data.get("community") or {}
                        have = main_community.get("have", release_have)
                        want = main_community.get("want", release_want)
                except Exception as exc:
                    logger.debug("Master %s lookup failed: %s", master_id, exc)

            df.at[idx, "have_count"] = have
            df.at[idx, "want_count"] = want

        except Exception as exc:
            logger.warning("Release %s detail failed: %s", rid, exc)

        if (i + 1) % 25 == 0:
            logger.info("Enriched have/want: %s / %s", i + 1, len(subset))

    return df


def _build_profile_affinities(
    conn: sqlite3.Connection,
    profile: list[dict],
    config: AppConfig,
) -> tuple[dict[str, float], dict[str, float], dict[int, float]]:
    """Compute normalized style, country, and year affinity dicts from the profile.

    Each dict maps a value to a score in [0, 1] where 1.0 = most frequent in the
    wantlist/collection (wantlist weighted at wantlist_weight, collection at
    collection_weight). Year affinities are smoothed ±2 years so nearby years
    share influence.

    Returns:
        style_aff   — {style_name: 0..1}
        country_aff — {country: 0..1}
        year_aff    — {year_int: 0..1}
    """
    pc = config.profile
    weight_map: dict[int, float] = {
        int(row["release_id"]): (
            pc.collection_weight if row.get("source") == "collection" else pc.wantlist_weight
        )
        for row in profile
    }
    release_ids = list(weight_map)
    if not release_ids:
        return {}, {}, {}

    batch_size = config.discover.sql_batch_size
    country_raw: dict[str, float] = defaultdict(float)
    year_raw: dict[int, float] = defaultdict(float)
    style_raw: dict[str, float] = defaultdict(float)

    for i in range(0, len(release_ids), batch_size):
        batch = release_ids[i : i + batch_size]
        p = ",".join("?" * len(batch))
        for rid, country, released in conn.execute(
            f"SELECT id, country, released FROM release WHERE id IN ({p})", batch
        ).fetchall():
            w = weight_map.get(int(rid), 1.0)
            if country:
                country_raw[str(country).strip()] += w
            year_str = str(released or "")[:4]
            try:
                year = int(year_str)
                if 1950 <= year <= 2030:
                    for offset in range(-2, 3):
                        decay = 1.0 - abs(offset) * 0.2
                        year_raw[year + offset] += w * decay
            except ValueError:
                pass

        for rid, style in conn.execute(
            f"SELECT release_id, style FROM release_style WHERE release_id IN ({p})", batch
        ).fetchall():
            w = weight_map.get(int(rid), 1.0)
            if style:
                style_raw[str(style).strip()] += w

    def _normalize(d: dict) -> dict:
        if not d:
            return {}
        max_val = max(d.values())
        return {k: v / max_val for k, v in d.items()} if max_val > 0 else dict(d)

    return _normalize(style_raw), _normalize(country_raw), _normalize(year_raw)


def compute_final_score(
    df: pd.DataFrame,
    config: AppConfig,
    *,
    style_aff: dict[str, float] | None = None,
    country_aff: dict[str, float] | None = None,
    year_aff: dict[int, float] | None = None,
    user_prefs: UserPreferences | None = None,
) -> pd.DataFrame:
    sc = config.score
    df = df.copy()
    base = df["metadata_score"].fillna(0).astype(float)

    have = df.get("have_count", pd.Series(dtype=float)).fillna(0).astype(float)
    want = df.get("want_count", pd.Series(dtype=float)).fillna(0).astype(float)
    if not isinstance(have, pd.Series):
        have = pd.Series(have, index=df.index)
    if not isinstance(want, pd.Series):
        want = pd.Series(want, index=df.index)

    # have_count/want_count are master-level (set by enrich_have_want)
    desirability = np.log1p(want) - np.log1p(have + 1.0)
    overknown_penalty = np.log1p(np.maximum(have - sc.overknown_threshold, 0)) / np.log1p(5000)

    # Style affinity — max affinity across the release's styles
    if style_aff and "styles" in df.columns:
        def _style_score(val: object) -> float:
            return max((style_aff.get(s, 0.0) for s in _parse_list_column(val)), default=0.0)
        style_boost = df["styles"].map(_style_score) * sc.style_affinity_weight
    else:
        style_boost = pd.Series(0.0, index=df.index)

    # Country affinity
    if country_aff and "country" in df.columns:
        country_boost = (
            df["country"].astype(str).str.strip().map(lambda c: country_aff.get(c, 0.0))
            * sc.country_affinity_weight
        )
    else:
        country_boost = pd.Series(0.0, index=df.index)

    # Year affinity
    if year_aff and "released" in df.columns:
        release_year = pd.to_numeric(df["released"].astype(str).str[:4], errors="coerce")
        year_boost = release_year.map(
            lambda y: year_aff.get(int(y), 0.0) if pd.notna(y) else 0.0
        ) * sc.year_affinity_weight
    else:
        year_boost = pd.Series(0.0, index=df.index)

    # Explicit user preference boosts — stack additively on top of learned affinities
    if user_prefs is not None and not user_prefs.is_empty():
        strength = user_prefs.boost_strength
        if user_prefs.preferred_styles and "styles" in df.columns:
            pref_style_set = set(user_prefs.preferred_styles)
            style_boost = style_boost + df["styles"].map(
                lambda val: strength if set(_parse_list_column(val)) & pref_style_set else 0.0
            )
        if user_prefs.preferred_countries and "country" in df.columns:
            pref_country_set = set(user_prefs.preferred_countries)
            country_boost = country_boost + df["country"].astype(str).str.strip().map(
                lambda c: strength if c in pref_country_set else 0.0
            )
        if user_prefs.year_from is not None and user_prefs.year_to is not None and "released" in df.columns:
            if "release_year" not in dir():
                release_year = pd.to_numeric(df["released"].astype(str).str[:4], errors="coerce")
            year_boost = year_boost + release_year.map(
                lambda y: strength if pd.notna(y) and user_prefs.year_from <= int(y) <= user_prefs.year_to else 0.0
            )

    df["desirability"] = desirability
    df["overknown_penalty"] = overknown_penalty
    df["style_boost"] = style_boost
    df["country_boost"] = country_boost
    df["year_boost"] = year_boost
    df["score"] = (
        sc.metadata_score_weight * base
        + sc.desirability_weight * desirability
        - sc.overknown_penalty_weight * overknown_penalty
        + style_boost
        + country_boost
        + year_boost
    )
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def score_candidates(
    config: AppConfig,
    *,
    force: bool = False,
    user_prefs: UserPreferences | None = None,
) -> Path:
    """Read candidates CSV, filter, rank, write recommendations."""
    in_path = config.paths.candidates_file
    out_path = config.paths.recommendations_file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.is_file():
        raise FileNotFoundError(
            f"Candidates not found at {in_path}. Run: python main.py discover"
        )

    if out_path.is_file() and not force:
        logger.info(
            "Recommendations exist at %s (use --force to rebuild)", out_path
        )
        return out_path

    df = pd.read_csv(in_path)
    logger.info("Loaded %s candidates", len(df))

    conn = sqlite3.connect(config.paths.db_path)
    try:
        profile_path = config.paths.profile_releases
        if profile_path.is_file():
            profile = load_profile_releases(profile_path)
            owned_releases, owned_masters = load_owned_sets(profile, conn)
            before_owned = len(df)
            df = filter_owned_releases(df, owned_releases, owned_masters, conn=conn)
            if before_owned != len(df):
                logger.info(
                    "Excluded %s already-owned candidates (%s → %s)",
                    before_owned - len(df),
                    before_owned,
                    len(df),
                )
        else:
            logger.warning(
                "Profile missing at %s; cannot exclude owned releases", profile_path
            )

        before = len(df)
        df = filter_compilations(df)
        logger.info("After compilation filter: %s → %s", before, len(df))

        style_aff, country_aff, year_aff = (
            _build_profile_affinities(conn, profile, config)
            if profile_path.is_file()
            else ({}, {}, {})
        )
        logger.info(
            "Profile affinities — styles: %s | countries: %s | years: %s",
            len(style_aff), len(country_aff), len(year_aff),
        )

        score_kwargs = dict(style_aff=style_aff, country_aff=country_aff, year_aff=year_aff, user_prefs=user_prefs)
        df = compute_final_score(df, config, **score_kwargs)
        primary_artist_ids = {
            int(ids[0])
            for val in df["artist_ids"]
            for ids in [_parse_artist_ids(val)]
            if ids
        }
        alias_map = _build_alias_canonical_map(conn, primary_artist_ids)
        logger.info(
            "Alias map built: %s artists → %s canonical groups",
            len(primary_artist_ids),
            len(set(alias_map.values())),
        )
        top = build_final_recommendations(df, config, conn, alias_map=alias_map)
        logger.info("Selected %s final recommendations", len(top))

        try:
            top = enrich_have_want(top, config)
        except ValueError as exc:
            logger.warning("Skipping have/want enrichment: %s", exc)

        top = compute_final_score(top, config, **score_kwargs)
        if user_prefs and not user_prefs.is_empty():
            logger.info(
                "Applied explicit preferences: %s style(s), %s country(ies), year %s–%s, boost +%.1f",
                len(user_prefs.preferred_styles),
                len(user_prefs.preferred_countries),
                user_prefs.year_from,
                user_prefs.year_to,
                user_prefs.boost_strength,
            )
    finally:
        conn.close()

    export_cols = [
        c
        for c in [
            "release_id",
            "title",
            "artists",
            "labels",
            "styles",
            "country",
            "released",
            "metadata_score",
            "style_boost",
            "country_boost",
            "year_boost",
            "have_count",
            "want_count",
            "release_have_count",
            "release_want_count",
            "desirability",
            "overknown_penalty",
            "score",
        ]
        if c in top.columns
    ]
    top[export_cols].to_csv(out_path, index=False)
    logger.info("Wrote top %s recommendations to %s", len(top), out_path)
    return out_path
