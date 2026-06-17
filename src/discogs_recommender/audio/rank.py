"""Re-rank top-100 metadata recommendations using audio embedding similarity."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from discogs_recommender.config import AppConfig
from discogs_recommender.audio.fetch import fetch_and_download
from discogs_recommender.audio.embed import (
    compute_candidate_embeddings,
    load_embedding_cache,
    save_embedding_cache,
)

logger = logging.getLogger(__name__)


def _load_profile_centroids(embeddings_path: Path, n_clusters: int = 1) -> np.ndarray:
    """Return profile centroids as a (k, dim) float32 array.

    When n_clusters=1 returns the mean centroid.
    When n_clusters>1 runs K-means on the raw per-track embeddings so each
    sub-genre cluster in the collection gets its own centroid. Candidates are
    then scored against the nearest one (max cosine similarity).
    """
    if not embeddings_path.is_file():
        raise FileNotFoundError(
            f"Collection embeddings not found at {embeddings_path}. "
            "Run the collection embedding step first to generate this file."
        )
    data = np.load(embeddings_path)
    embs = data["embeddings"].astype(np.float32) if "embeddings" in data else data[list(data.keys())[0]].astype(np.float32)

    k = min(n_clusters, len(embs))

    if k <= 1:
        centroid = np.mean(embs, axis=0, keepdims=True)
        logger.info("Single collection centroid (dim=%s, tracks=%s)", centroid.shape[1], len(embs))
        return centroid

    try:
        from sklearn.cluster import KMeans  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("scikit-learn is required for multi-centroid mode") from exc

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(embs)
    centroids = km.cluster_centers_.astype(np.float32)
    sizes = np.bincount(km.labels_).tolist()
    logger.info("K-means profile: %s clusters from %s tracks (sizes: %s)", k, len(embs), sizes)
    return centroids


def _max_cosine_similarity(emb: np.ndarray, centroids: np.ndarray) -> float:
    """Return the highest cosine similarity between emb and any centroid row."""
    emb_norm = np.linalg.norm(emb)
    if emb_norm == 0:
        return 0.0
    centroid_norms = np.linalg.norm(centroids, axis=1)
    valid = centroid_norms > 0
    if not np.any(valid):
        return 0.0
    sims = (centroids[valid] @ emb) / (centroid_norms[valid] * emb_norm)
    return float(np.max(sims))


def _normalize_series(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def audio_rank(
    df: pd.DataFrame,
    config: AppConfig,
    *,
    profile_path: Path | None = None,
) -> pd.DataFrame:
    """Take the top-100 recommendations and return the top-N refined by audio.

    Steps:
      1. Download audio for each release (yt-dlp from Discogs video URL).
      2. Compute EffNet embeddings.
      3. Score each candidate by cosine similarity to the collection centroid.
      4. Blend with normalised metadata score and return top *top_n_final* rows.
    """
    ac = config.audio

    release_ids = df["release_id"].astype(int).tolist()

    # Always fetch audio (skips download for already-cached files); ensures url_map is complete
    audio_paths, url_map = fetch_and_download(release_ids, config)

    if not audio_paths:
        logger.warning(
            "No audio downloaded; falling back to metadata score for top-%s", ac.top_n_final
        )
        return df.head(ac.top_n_final).copy()

    # Load persisted embedding cache; only run EffNet on releases not yet cached
    cache_path = config.paths.candidate_embeddings_cache
    emb_cache = load_embedding_cache(cache_path)
    to_embed = {rid: paths for rid, paths in audio_paths.items() if rid not in emb_cache}

    logger.info(
        "Embedding cache: %s hits, %s to embed (cache at %s)",
        len(audio_paths) - len(to_embed), len(to_embed), cache_path,
    )

    if to_embed:
        new_embeddings = compute_candidate_embeddings(to_embed, config.paths.effnet_model)
        emb_cache.update(new_embeddings)
        save_embedding_cache(cache_path, emb_cache)

    embeddings = {rid: emb_cache[rid] for rid in release_ids if rid in emb_cache}

    active_profile = profile_path or config.paths.collection_embeddings
    if profile_path:
        logger.info("Using custom vibe profile: %s", profile_path)
    profile_centroids = _load_profile_centroids(active_profile, n_clusters=ac.n_clusters)

    # Verify dimension compatibility
    sample_emb = next(iter(embeddings.values()))
    if sample_emb.shape[0] != profile_centroids.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: candidate={sample_emb.shape[0]} "
            f"vs profile={profile_centroids.shape[1]}. "
            "Ensure you used the same EffNet model to build both."
        )

    df = df.copy()
    df["audio_sim"] = df["release_id"].map(
        lambda rid: _max_cosine_similarity(embeddings[int(rid)], profile_centroids)
        if int(rid) in embeddings
        else 0.0
    )

    n_audio = (df["audio_sim"] > 0).sum()
    logger.info(
        "Audio similarity computed for %s / %s releases", n_audio, len(df)
    )

    meta_col = "score" if "score" in df.columns else "metadata_score"
    norm_meta = _normalize_series(df[meta_col].fillna(0).astype(float))
    norm_audio = _normalize_series(df["audio_sim"])
    df["audio_blend_score"] = ac.audio_weight * norm_audio + (1.0 - ac.audio_weight) * norm_meta

    # Hard 50/50 bucket split: half from affinity (style/label/artist matched),
    # half from discovery (style-only, surfaces unknown labels). Each half is
    # ranked independently by blend score so the best of each bucket wins.
    bucket_col = df.get("bucket", pd.Series("affinity", index=df.index))
    if not isinstance(bucket_col, pd.Series):
        bucket_col = pd.Series("affinity", index=df.index)

    n_each = ac.top_n_final // 2
    affinity_top = (
        df[bucket_col != "discovery"]
        .sort_values("audio_blend_score", ascending=False)
        .head(n_each)
    )
    discovery_top = (
        df[bucket_col == "discovery"]
        .sort_values("audio_blend_score", ascending=False)
        .head(ac.top_n_final - n_each)
    )
    logger.info(
        "Bucket split — affinity: %s | discovery: %s",
        len(affinity_top), len(discovery_top),
    )

    top = (
        pd.concat([affinity_top, discovery_top])
        .sort_values("audio_sim", ascending=False)
        .reset_index(drop=True)
    )

    # Add convenience URL columns
    top["discogs_url"] = top["release_id"].map(
        lambda rid: f"https://www.discogs.com/release/{int(rid)}"
    )
    top["youtube_url"] = top["release_id"].map(
        lambda rid: (url_map.get(int(rid)) or [""])[0]
    )

    logger.info(
        "Audio-refined top %s selected (audio_weight=%.2f)",
        len(top),
        ac.audio_weight,
    )
    return top


def _load_recent_recommendation_ids(exports_dir: Path, n_runs: int) -> set[int]:
    """Return release_ids that appeared in the last *n_runs* versioned top10 files."""
    if n_runs <= 0:
        return set()
    top10_dir = exports_dir / "top10"
    if not top10_dir.is_dir():
        return set()
    existing = sorted(top10_dir.glob("top10_v*.csv"))
    recent = existing[-n_runs:]
    seen: set[int] = set()
    for path in recent:
        try:
            df = pd.read_csv(path, usecols=["release_id"])
            seen.update(df["release_id"].dropna().astype(int).tolist())
        except Exception as exc:
            logger.debug("Could not read history file %s: %s", path, exc)
    return seen


def _next_top10_path(exports_dir: Path) -> Path:
    """Return the next versioned path, e.g. exports/top10/top10_v3.csv."""
    top10_dir = exports_dir / "top10"
    top10_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(top10_dir.glob("top10_v*.csv"))
    if not existing:
        next_v = 1
    else:
        import re
        nums = [int(m.group(1)) for f in existing if (m := re.search(r"top10_v(\d+)\.csv", f.name))]
        next_v = max(nums) + 1 if nums else 1
    return top10_dir / f"top10_v{next_v}.csv"


def run_audio_rank(config: AppConfig, *, force: bool = False, profile_path: Path | None = None) -> Path:
    """Read recommendations.csv, run audio ranking, write a versioned top10 CSV."""
    in_path = config.paths.recommendations_file

    if not in_path.is_file():
        raise FileNotFoundError(
            f"Recommendations not found at {in_path}. Run: python main.py score"
        )

    out_path = _next_top10_path(config.paths.exports_dir)

    df = pd.read_csv(in_path)
    logger.info("Loaded %s recommendations for audio ranking", len(df))

    n_history = config.audio.history_exclusion_runs
    if n_history > 0:
        recent_ids = _load_recent_recommendation_ids(config.paths.exports_dir, n_history)
        if recent_ids:
            before = len(df)
            df = df[~df["release_id"].astype(int).isin(recent_ids)].reset_index(drop=True)
            excluded = before - len(df)
            logger.info(
                "History exclusion (last %s runs): removed %s already-recommended releases "
                "(%s → %s candidates)",
                n_history, excluded, before, len(df),
            )
            if len(df) < config.audio.top_n_final:
                logger.warning(
                    "Only %s candidates remain after history exclusion — "
                    "reduce history_exclusion_runs or run score stage again",
                    len(df),
                )

    top = audio_rank(df, config, profile_path=profile_path)

    export_cols = [
        c for c in [
            "release_id", "bucket", "title", "artists", "labels", "styles",
            "country", "released", "score", "metadata_score",
            "audio_sim", "audio_blend_score",
            "have_count", "want_count",
            "image_url", "discogs_url", "youtube_url",
        ]
        if c in top.columns
    ]
    top[export_cols].to_csv(out_path, index=False)
    logger.info("Wrote top %s audio-ranked recommendations to %s", len(top), out_path)
    return out_path
