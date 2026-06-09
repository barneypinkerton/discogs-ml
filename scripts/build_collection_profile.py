"""Build EffNet taste-profile embeddings from your local music collection.

Usage:
    python scripts/build_collection_profile.py --mp3-dir /path/to/music [--force]

Reads every .mp3/.wav/.flac/.aiff/.m4a file in the given directory, runs
Essentia EffNet on each, mean-pools frame embeddings into one vector per track,
computes a centroid across all tracks, and writes:

    $DISCOGS_DATA_ROOT/embeddings/my_collection_embeddings.npz

Run this once, then re-run `python main.py audio_rank --force` to pick up
the updated profile whenever you add new tracks.

Prerequisites:
    pip install -r requirements-audio.txt
    # Download discogs_artist_embeddings-effnet-bs64-1.pb from
    # https://essentia.upf.edu/models.html and place in $DISCOGS_DATA_ROOT/embeddings/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Defaults — all overridable via CLI flags or DISCOGS_DATA_ROOT env var
# ---------------------------------------------------------------------------
_DATA_ROOT = Path(
    __import__("os").environ.get("DISCOGS_DATA_ROOT", str(Path.home() / "DiscogsData"))
).expanduser()

DEFAULT_MP3_DIR: Path | None = None  # no default — user must supply --mp3-dir
DEFAULT_MODEL_PATH = _DATA_ROOT / "embeddings" / "discogs_artist_embeddings-effnet-bs64-1.pb"
DEFAULT_SAVE_PATH = _DATA_ROOT / "embeddings" / "my_collection_embeddings.npz"

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aiff", ".aif", ".m4a"}
SAMPLE_RATE = 16_000


def _load_essentia():
    try:
        from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs
        return MonoLoader, TensorflowPredictEffnetDiscogs
    except ImportError:
        print("ERROR: essentia-tensorflow not installed.")
        print("  pip install -r requirements-audio.txt")
        sys.exit(1)


def embed_file(path: Path, model, MonoLoader) -> np.ndarray | None:
    try:
        audio = MonoLoader(filename=str(path), sampleRate=SAMPLE_RATE, resampleQuality=4)()
        emb = model(audio)
        return emb.mean(axis=0).astype(np.float32)
    except Exception as exc:
        print(f"  SKIP {path.name}: {exc}")
        return None


def build_profile(
    mp3_dir: Path,
    model_path: Path,
    save_path: Path,
    *,
    force: bool = False,
) -> Path:
    if save_path.exists() and not force:
        print(f"Profile already exists at {save_path}")
        print("Use --force to rebuild.")
        return save_path

    if not model_path.is_file():
        print(f"ERROR: EffNet model not found at {model_path}")
        print("Download from: https://essentia.upf.edu/models.html")
        print("  → Audio classification → Discogs-EffNet")
        sys.exit(1)

    audio_files = sorted(
        p for p in mp3_dir.iterdir()
        if p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        print(f"No audio files found in {mp3_dir}")
        sys.exit(1)

    print(f"Found {len(audio_files)} audio files in {mp3_dir}")
    print(f"Loading EffNet model from {model_path} …")

    MonoLoader, TensorflowPredictEffnetDiscogs = _load_essentia()
    model = TensorflowPredictEffnetDiscogs(
        graphFilename=str(model_path),
        output="PartitionedCall:1",
    )

    filenames: list[str] = []
    embeddings: list[np.ndarray] = []
    failed = 0

    for i, path in enumerate(audio_files, 1):
        emb = embed_file(path, model, MonoLoader)
        if emb is not None:
            filenames.append(path.name)
            embeddings.append(emb)
        else:
            failed += 1

        if i % 25 == 0 or i == len(audio_files):
            ok = len(embeddings)
            print(f"  [{i}/{len(audio_files)}] embedded {ok} tracks ({failed} failed)")

    if not embeddings:
        print("ERROR: No tracks embedded successfully.")
        sys.exit(1)

    emb_matrix = np.array(embeddings, dtype=np.float32)
    centroid = emb_matrix.mean(axis=0).reshape(1, -1)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,
        filenames=filenames,
        embeddings=emb_matrix,
        centroid=centroid,
    )

    print(f"\nProfile built from {len(embeddings)} tracks ({failed} skipped)")
    print(f"Centroid shape: {centroid.shape}")
    print(f"Saved to: {save_path}")

    # Sanity check: most and least representative tracks
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    sims = cos_sim(emb_matrix, centroid).ravel()
    ranked = sorted(zip(filenames, sims), key=lambda x: x[1], reverse=True)

    print("\nMost representative tracks (highest cosine similarity to centroid):")
    for name, sim in ranked[:5]:
        print(f"  {sim:.4f}  {name}")

    print("\nLeast representative tracks (potential outliers):")
    for name, sim in ranked[-5:]:
        print(f"  {sim:.4f}  {name}")

    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build EffNet collection profile from local MP3s",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mp3-dir",
        type=Path,
        required=True,
        help="Directory containing your MP3/WAV/FLAC collection",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to discogs_artist_embeddings-effnet-bs64-1.pb",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SAVE_PATH,
        help="Where to write my_collection_embeddings.npz",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if output file already exists",
    )
    args = parser.parse_args()

    mp3_dir = args.mp3_dir.expanduser().resolve()
    if not mp3_dir.is_dir():
        print(f"ERROR: --mp3-dir does not exist: {mp3_dir}")
        sys.exit(1)
    build_profile(mp3_dir, args.model, args.output, force=args.force)


if __name__ == "__main__":
    main()
