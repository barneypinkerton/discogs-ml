"""Compute Essentia EffNet embeddings from audio files."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _load_essentia_model(model_path: Path):
    try:
        import essentia.standard as es  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "essentia-tensorflow is required for audio embeddings. "
            "Install it with: pip install -r requirements-audio.txt"
        ) from exc

    if not model_path.is_file():
        raise FileNotFoundError(
            f"EffNet model not found at {model_path}. "
            "Download the Discogs EffNet model from https://essentia.upf.edu/models.html "
            "and place it at that path."
        )

    return es.TensorflowPredictEffnetDiscogs(
        graphFilename=str(model_path),
        output="PartitionedCall:1",
    )


def embed_audio_file(
    audio_path: Path,
    model,
) -> np.ndarray | None:
    """Return mean-pooled EffNet embedding for *audio_path*, or None on error.

    The mean pool is over the temporal frames produced by the EffNet model for
    a single audio file.
    """
    try:
        import essentia.standard as es  # type: ignore[import]

        loader = es.MonoLoader(
            filename=str(audio_path),
            sampleRate=16000,
            resampleQuality=4,
        )
        audio = loader()
        if len(audio) == 0:
            logger.warning("Empty audio loaded from %s", audio_path)
            return None

        embeddings = model(audio)
        if embeddings is None or len(embeddings) == 0:
            logger.warning("No embeddings produced for %s", audio_path)
            return None

        return np.mean(embeddings, axis=0).astype(np.float32)

    except Exception as exc:
        logger.warning("Embedding failed for %s: %s", audio_path, exc)
        return None


def compute_candidate_embeddings(
    audio_paths: dict[int, list[Path]],
    model_path: Path,
) -> dict[int, np.ndarray]:
    """Compute a single embedding per release by averaging across all its tracks.

    For each release, every downloaded track is embedded individually and the
    results are mean-pooled into one release-level vector.  This means a record
    with 3 consistently decent tracks will score similarly to one with 3 decent
    tracks, whereas a record with 1 great track + 2 poor ones will score lower
    than it would if only the great track were considered.

    Returns a dict of release_id → embedding vector (1-D float32 array).
    """
    if not audio_paths:
        return {}

    model = _load_essentia_model(model_path)
    results: dict[int, np.ndarray] = {}
    releases_done = 0

    for rid, paths in audio_paths.items():
        track_embeddings: list[np.ndarray] = []
        for path in paths:
            emb = embed_audio_file(path, model)
            if emb is not None:
                track_embeddings.append(emb)

        if track_embeddings:
            results[rid] = np.mean(track_embeddings, axis=0).astype(np.float32)
            logger.debug(
                "Release %s: %s / %s tracks embedded successfully",
                rid,
                len(track_embeddings),
                len(paths),
            )

        releases_done += 1
        if releases_done % 10 == 0:
            logger.info("Embedded: %s / %s releases", releases_done, len(audio_paths))

    logger.info(
        "Embeddings computed: %s / %s releases",
        len(results),
        len(audio_paths),
    )
    return results
