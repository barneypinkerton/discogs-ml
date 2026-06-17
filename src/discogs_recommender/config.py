"""Load configuration from YAML and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "config" / "default.yaml"


def _expand(path: str | Path, data_root: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return data_root / p


@dataclass
class PathsConfig:
    data_root: Path
    csv_dir: Path
    db_path: Path
    labels_xml: Path
    catalog_dir: Path
    label_data_file: Path
    label_family_file: Path
    embeddings_dir: Path
    collection_embeddings: Path
    candidate_embeddings_cache: Path
    effnet_model: Path
    exports_dir: Path
    profile_dir: Path
    profile_releases: Path
    candidates_file: Path
    recommendations_file: Path
    top10_file: Path
    candidate_audio_dir: Path


@dataclass
class DiscogsApiConfig:
    user_token: str
    username: str
    requests_per_min: int = 30
    per_page: int = 50
    timeout_s: int = 40
    max_retries: int = 7


@dataclass
class ProfileConfig:
    collection_weight: float = 1.0
    wantlist_weight: float = 1.5


@dataclass
class DiscoverConfig:
    family_decay: float = 0.85
    sibling_decay_factor: float = 0.7
    artist_to_label_decay: float = 0.7
    label_to_artist_decay: float = 0.5
    top_artists_for_expansion: int = 10_000
    top_labels_for_expansion: int = 1_500
    candidate_pool_limit: int = 50_000
    compilation_pool_fraction: float = 0.10
    discovery_pool_fraction: float = 0.45
    min_style_match_count: int = 2
    genre: str = "Electronic"
    sql_batch_size: int = 500


@dataclass
class AudioConfig:
    top_n_final: int = 10
    audio_weight: float = 0.6
    cache_audio: bool = True
    download_timeout_s: int = 120
    max_videos_per_release: int = 6
    download_workers: int = 4
    history_exclusion_runs: int = 1
    n_clusters: int = 4


@dataclass
class ScoreConfig:
    max_per_artist: int = 2
    max_per_label: int = 3
    top_n_export: int = 300
    enrich_have_want_limit: int = 300
    desirability_weight: float = 0.25
    overknown_penalty_weight: float = 0.6
    overknown_threshold: int = 500
    max_have_count: int = 600
    max_want_count: int = 600
    exclude_owned_artists: bool = True
    metadata_score_weight: float = 0.4
    style_affinity_weight: float = 3.0
    country_affinity_weight: float = 1.5
    year_affinity_weight: float = 1.5


@dataclass
class AppConfig:
    paths: PathsConfig
    discogs: DiscogsApiConfig
    profile: ProfileConfig
    discover: DiscoverConfig
    score: ScoreConfig
    audio: AudioConfig
    pipeline_default_stages: list[str] = field(default_factory=list)
    label_graph_parse_progress_every: int = 500_000
    repo_root: Path = field(default_factory=lambda: _REPO_ROOT)
    vendor_discogs_xml2db: Path = field(
        default_factory=lambda: _REPO_ROOT / "vendor" / "discogs-xml2db"
    )


def load_config(config_path: Path | None = None) -> AppConfig:
    load_dotenv(_REPO_ROOT / ".env")
    config_path = config_path or _DEFAULT_CONFIG
    with open(config_path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    data_root = Path(
        os.environ.get("DISCOGS_DATA_ROOT", str(Path.home() / "DiscogsData"))
    ).expanduser()

    paths_raw = raw.get("paths", {})
    paths = PathsConfig(
        data_root=data_root,
        csv_dir=_expand(paths_raw.get("csv_dir", "csv"), data_root),
        db_path=_expand(
            os.environ.get("DISCOGS_DB_PATH", paths_raw.get("db_file", "db/discogs.sqlite")),
            data_root,
        ),
        labels_xml=_expand(
            os.environ.get(
                "DISCOGS_LABELS_XML",
                paths_raw.get("labels_xml", "xml/discogs_20250101_labels.xml"),
            ),
            data_root,
        ),
        catalog_dir=_expand(paths_raw.get("catalog_dir", "catalog"), data_root),
        label_data_file=_expand(
            paths_raw.get("label_data_file", "catalog/label_data.json"), data_root
        ),
        label_family_file=_expand(
            paths_raw.get("label_family_file", "catalog/label_family.json"), data_root
        ),
        embeddings_dir=_expand(paths_raw.get("embeddings_dir", "embeddings"), data_root),
        collection_embeddings=_expand(
            paths_raw.get(
                "collection_embeddings", "embeddings/my_collection_embeddings.npz"
            ),
            data_root,
        ),
        candidate_embeddings_cache=_expand(
            paths_raw.get(
                "candidate_embeddings_cache",
                "embeddings/candidate_embeddings_cache.npz",
            ),
            data_root,
        ),
        effnet_model=_expand(
            paths_raw.get(
                "effnet_model", "embeddings/discogs_artist_embeddings-effnet-bs64-1.pb"
            ),
            data_root,
        ),
        exports_dir=_expand(paths_raw.get("exports_dir", "exports"), data_root),
        profile_dir=_expand(paths_raw.get("profile_dir", "profile"), data_root),
        profile_releases=_expand(
            paths_raw.get("profile_releases", "profile/releases.json"), data_root
        ),
        candidates_file=_expand(
            paths_raw.get("candidates_file", "exports/candidates.csv"), data_root
        ),
        recommendations_file=_expand(
            paths_raw.get("recommendations_file", "exports/recommendations.csv"),
            data_root,
        ),
        top10_file=_expand(
            paths_raw.get("top10_file", "exports/top10.csv"),
            data_root,
        ),
        candidate_audio_dir=_expand(
            paths_raw.get("candidate_audio_dir", "candidate_audio"), data_root
        ),
    )

    discogs_raw = raw.get("discogs", {})
    profile_raw = raw.get("profile", {})
    discover_raw = raw.get("discover", {})
    score_raw = raw.get("score", {})
    audio_raw = raw.get("audio", {})
    pipeline = raw.get("pipeline", {})
    label_graph = raw.get("label_graph", {})

    token = os.environ.get("DISCOGS_USER_TOKEN", "").strip()
    username = os.environ.get("DISCOGS_USERNAME", "").strip()

    return AppConfig(
        paths=paths,
        discogs=DiscogsApiConfig(
            user_token=token,
            username=username,
            requests_per_min=int(discogs_raw.get("requests_per_min", 30)),
            per_page=int(discogs_raw.get("per_page", 50)),
            timeout_s=int(discogs_raw.get("timeout_s", 40)),
            max_retries=int(discogs_raw.get("max_retries", 7)),
        ),
        profile=ProfileConfig(
            collection_weight=float(profile_raw.get("collection_weight", 1.0)),
            wantlist_weight=float(profile_raw.get("wantlist_weight", 1.5)),
        ),
        discover=DiscoverConfig(
            family_decay=float(discover_raw.get("family_decay", 0.85)),
            sibling_decay_factor=float(
                discover_raw.get("sibling_decay_factor", 0.7)
            ),
            artist_to_label_decay=float(
                discover_raw.get("artist_to_label_decay", 0.7)
            ),
            label_to_artist_decay=float(
                discover_raw.get("label_to_artist_decay", 0.5)
            ),
            top_artists_for_expansion=int(
                discover_raw.get("top_artists_for_expansion", 100)
            ),
            top_labels_for_expansion=int(
                discover_raw.get("top_labels_for_expansion", 80)
            ),
            candidate_pool_limit=int(
                discover_raw.get("candidate_pool_limit", 50_000)
            ),
            compilation_pool_fraction=float(
                discover_raw.get("compilation_pool_fraction", 0.10)
            ),
            discovery_pool_fraction=float(
                discover_raw.get("discovery_pool_fraction", 0.45)
            ),
            min_style_match_count=int(discover_raw.get("min_style_match_count", 2)),
            genre=str(discover_raw.get("genre", "Electronic")),
            sql_batch_size=int(discover_raw.get("sql_batch_size", 500)),
        ),
        score=ScoreConfig(
            max_per_artist=int(score_raw.get("max_per_artist", 3)),
            max_per_label=int(score_raw.get("max_per_label", 5)),
            top_n_export=int(score_raw.get("top_n_export", 300)),
            enrich_have_want_limit=int(score_raw.get("enrich_have_want_limit", 300)),
            max_have_count=int(score_raw.get("max_have_count", 600)),
            max_want_count=int(score_raw.get("max_want_count", 600)),
            exclude_owned_artists=bool(score_raw.get("exclude_owned_artists", True)),
            desirability_weight=float(score_raw.get("desirability_weight", 0.25)),
            overknown_penalty_weight=float(
                score_raw.get("overknown_penalty_weight", 0.6)
            ),
            overknown_threshold=int(score_raw.get("overknown_threshold", 500)),
            metadata_score_weight=float(score_raw.get("metadata_score_weight", 0.4)),
            style_affinity_weight=float(score_raw.get("style_affinity_weight", 3.0)),
            country_affinity_weight=float(score_raw.get("country_affinity_weight", 1.5)),
            year_affinity_weight=float(score_raw.get("year_affinity_weight", 1.5)),
        ),
        audio=AudioConfig(
            top_n_final=int(audio_raw.get("top_n_final", 10)),
            audio_weight=float(audio_raw.get("audio_weight", 0.6)),
            cache_audio=bool(audio_raw.get("cache_audio", True)),
            download_timeout_s=int(audio_raw.get("download_timeout_s", 120)),
            max_videos_per_release=int(audio_raw.get("max_videos_per_release", 6)),
            download_workers=int(audio_raw.get("download_workers", 4)),
            history_exclusion_runs=int(audio_raw.get("history_exclusion_runs", 3)),
            n_clusters=int(audio_raw.get("n_clusters", 4)),
        ),
        pipeline_default_stages=list(
            pipeline.get(
                "default_stages",
                ["build_labels", "sync_profile", "discover", "score"],
            )
        ),
        label_graph_parse_progress_every=int(
            label_graph.get("parse_progress_every", 500_000)
        ),
    )
