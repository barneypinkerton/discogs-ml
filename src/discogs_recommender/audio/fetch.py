"""Fetch video URLs from Discogs API and download audio via yt-dlp."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Prefer the Homebrew yt-dlp (newer, Python-version-independent) over any
# pip-installed version that may be pinned to an older YouTube extractor.
_YTDLP_CANDIDATES = ["/opt/homebrew/bin/yt-dlp", "/usr/local/bin/yt-dlp"]
_YTDLP_BIN: str = next(
    (p for p in _YTDLP_CANDIDATES if Path(p).exists()),
    shutil.which("yt-dlp") or "yt-dlp",
)

from discogs_recommender.config import AppConfig
from discogs_recommender.profile.discogs_client import DiscogsClient

logger = logging.getLogger(__name__)


def _get_video_urls(release_data: dict[str, Any]) -> list[str]:
    """Return all YouTube URLs from a release's videos list."""
    return [
        video["uri"]
        for video in (release_data.get("videos") or [])
        if "youtube.com" in video.get("uri", "") or "youtu.be" in video.get("uri", "")
    ]


def fetch_video_urls(
    release_ids: list[int],
    client: DiscogsClient,
    max_per_release: int = 6,
) -> dict[int, list[str]]:
    """Return all YouTube URLs for each release_id (capped at max_per_release)."""
    result: dict[int, list[str]] = {}
    for i, rid in enumerate(release_ids):
        try:
            data = client.get(
                f"https://api.discogs.com/releases/{rid}",
                what=f"release_{rid}",
            )
            result[rid] = _get_video_urls(data)[:max_per_release]
        except Exception as exc:
            logger.warning("release %s video lookup failed: %s", rid, exc)
            result[rid] = []
        if (i + 1) % 10 == 0:
            logger.info("Fetched video URLs: %s / %s", i + 1, len(release_ids))
    return result


def download_audio(
    release_id: int,
    url: str,
    audio_dir: Path,
    *,
    track_idx: int = 0,
    timeout_s: int = 120,
    cache: bool = True,
) -> Path | None:
    """Download full audio for *url* to *audio_dir*/{release_id}_t{track_idx}.wav."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_path = audio_dir / f"{release_id}_t{track_idx}.wav"

    if cache and out_path.exists():
        logger.debug("Audio cache hit: %s", out_path)
        return out_path

    cmd = [
        _YTDLP_BIN,
        "--no-playlist",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "-o", str(out_path),
        "--quiet",
        url,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=timeout_s, capture_output=True)
        if out_path.exists():
            return out_path
        logger.warning("yt-dlp succeeded but %s not found", out_path)
    except subprocess.TimeoutExpired:
        logger.warning("yt-dlp timed out for release %s track %s", release_id, track_idx)
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "yt-dlp failed for release %s track %s (exit %s): %s",
            release_id, track_idx, exc.returncode,
            exc.stderr.decode(errors="replace")[:200],
        )
    return None


def fetch_and_download(
    release_ids: list[int],
    config: AppConfig,
) -> tuple[dict[int, list[Path]], dict[int, list[str]]]:
    """Fetch video URLs (cached) then download all tracks in parallel.

    Returns:
        audio_paths — release_id → list of local wav paths (one per track)
        url_map     — release_id → list of YouTube URLs (for linking in the UI)
    """
    client = DiscogsClient(config.discogs)
    audio_cfg = config.audio
    audio_dir = config.paths.candidate_audio_dir
    audio_dir.mkdir(parents=True, exist_ok=True)

    # ── URL cache ──────────────────────────────────────────────────────────
    url_cache_file = audio_dir / "url_cache.json"
    url_map: dict[int, list[str]] = {}

    if audio_cfg.cache_audio and url_cache_file.exists():
        try:
            raw = json.loads(url_cache_file.read_text())
            url_map = {int(k): v for k, v in raw.items()}
        except Exception:
            url_map = {}

    missing = [rid for rid in release_ids if rid not in url_map]
    if missing:
        logger.info("Fetching video URLs for %s releases …", len(missing))
        new_urls = fetch_video_urls(missing, client, audio_cfg.max_videos_per_release)
        url_map.update(new_urls)
        try:
            url_cache_file.write_text(json.dumps({str(k): v for k, v in url_map.items()}))
        except Exception as exc:
            logger.warning("Could not write URL cache: %s", exc)
    else:
        logger.info("Video URLs loaded from cache for all %s releases", len(release_ids))

    total_urls = sum(len(url_map.get(rid, [])) for rid in release_ids)
    releases_with_video = sum(1 for rid in release_ids if url_map.get(rid))
    logger.info(
        "Video URLs: %s tracks across %s / %s releases — downloading with %s workers …",
        total_urls, releases_with_video, len(release_ids), audio_cfg.download_workers,
    )

    # ── Parallel downloads ─────────────────────────────────────────────────
    audio_dict: dict[int, dict[int, Path]] = defaultdict(dict)
    tracks_attempted = 0

    with ThreadPoolExecutor(max_workers=audio_cfg.download_workers) as executor:
        futures: dict[Any, tuple[int, int]] = {}
        for rid in release_ids:
            for i, url in enumerate(url_map.get(rid, [])):
                f = executor.submit(
                    download_audio, rid, url, audio_dir,
                    track_idx=i,
                    timeout_s=audio_cfg.download_timeout_s,
                    cache=audio_cfg.cache_audio,
                )
                futures[f] = (rid, i)

        for future in as_completed(futures):
            rid, track_idx = futures[future]
            tracks_attempted += 1
            try:
                path = future.result()
                if path is not None:
                    audio_dict[rid][track_idx] = path
            except Exception as exc:
                logger.warning("Release %s track %s failed: %s", rid, track_idx, exc)

            if tracks_attempted % 10 == 0:
                done = sum(1 for d in audio_dict.values() if d)
                logger.info(
                    "Downloading: %s / %s tracks (%s releases done)",
                    tracks_attempted, total_urls, done,
                )

    audio_paths = {
        rid: [d[i] for i in sorted(d.keys())]
        for rid, d in audio_dict.items()
        if d
    }
    total_tracks = sum(len(p) for p in audio_paths.values())
    logger.info(
        "Audio downloaded: %s tracks across %s / %s releases",
        total_tracks, len(audio_paths), len(release_ids),
    )
    return audio_paths, url_map
