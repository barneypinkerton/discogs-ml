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
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_DATA_ROOT = Path(
    os.environ.get("DISCOGS_DATA_ROOT", str(Path.home() / "DiscogsData"))
).expanduser()

DEFAULT_MODEL_PATH = _DATA_ROOT / "embeddings" / "discogs_artist_embeddings-effnet-bs64-1.pb"
DEFAULT_SAVE_PATH = _DATA_ROOT / "embeddings" / "my_collection_embeddings.npz"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build EffNet collection profile from local audio files",
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

    from discogs_recommender.audio.embed import build_vibe_profile
    try:
        build_vibe_profile(mp3_dir, args.model, args.output, force=args.force)
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
