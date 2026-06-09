"""Re-rank top-100 metadata recommendations using audio embedding similarity."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from discogs_recommender.config import AppConfig
from discogs_recommender.audio.fetch import fetch_and_download
from discogs_recommender.audio.embed import compute_candidate_embeddings

logger = logging.getLogger(__name__)


def _load_profile_centroid(embeddings_path: Path) -> np.ndarray:
    """Return the collection taste-profile centroid vector.

    Prefers the pre-computed weighted 'centroid' key if present; falls back to
    the mean of the raw 'embeddings' array otherwise.
    """
    if not embeddings_path.is_file():
        raise FileNotFoundError(
            f"Collection embeddings not found at {embeddings_path}. "
            "Run the collection embedding step first to generate this file."
        )
    data = np.load(embeddings_path)
    if "centroid" in data:
        centroid = np.asarray(data["centroid"]).squeeze().astype(np.float32)
        logger.info("Loaded pre-computed collection centroid (dim=%s)", centroid.shape[0])
        return centroid
    if "embeddings" in data:
        embs = data["embeddings"]
    else:
        key = list(data.keys())[0]
        logger.debug("Using npz key '%s' as embeddings array", key)
        embs = data[key]
    centroid = np.mean(embs, axis=0).astype(np.float32)
    logger.info("Computed collection centroid from %s embeddings (dim=%s)", len(embs), centroid.shape[0])
    return centroid


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _normalize_series(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def audio_rank(
    df: pd.DataFrame,
    config: AppConfig,
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
    audio_paths, url_map = fetch_and_download(release_ids, config)

    if not audio_paths:
        logger.warning(
            "No audio downloaded; falling back to metadata score for top-%s", ac.top_n_final
        )
        return df.head(ac.top_n_final).copy()

    embeddings = compute_candidate_embeddings(audio_paths, config.paths.effnet_model)

    profile_centroid = _load_profile_centroid(config.paths.collection_embeddings)

    # Verify dimension compatibility
    sample_emb = next(iter(embeddings.values()))
    if sample_emb.shape != profile_centroid.shape:
        raise ValueError(
            f"Embedding dimension mismatch: candidate={sample_emb.shape} "
            f"vs profile={profile_centroid.shape}. "
            "Ensure you used the same EffNet model to build both."
        )

    df = df.copy()
    df["audio_sim"] = df["release_id"].map(
        lambda rid: _cosine_similarity(embeddings[int(rid)], profile_centroid)
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

    df["audio_blend_score"] = (
        ac.audio_weight * norm_audio + (1.0 - ac.audio_weight) * norm_meta
    )

    top = (
        df.sort_values("audio_blend_score", ascending=False)
        .head(ac.top_n_final)
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


def run_audio_rank(config: AppConfig, *, force: bool = False) -> Path:
    """Read recommendations.csv, run audio ranking, write top10.csv."""
    in_path = config.paths.recommendations_file
    out_path = config.paths.top10_file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.is_file() and not force:
        logger.info("Top-10 cache at %s (use --force to rebuild)", out_path)
        return out_path

    if not in_path.is_file():
        raise FileNotFoundError(
            f"Recommendations not found at {in_path}. Run: python main.py score"
        )

    df = pd.read_csv(in_path)
    logger.info("Loaded %s recommendations for audio ranking", len(df))

    top = audio_rank(df, config)

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
