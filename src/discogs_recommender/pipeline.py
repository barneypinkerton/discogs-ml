"""Orchestrate pipeline stages in order."""

from __future__ import annotations

import logging

from discogs_recommender.audio.rank import run_audio_rank
from discogs_recommender.catalog.label_graph import build_label_graph
from discogs_recommender.config import AppConfig
from discogs_recommender.profile.sync import sync_profile
from discogs_recommender.recommend.discover import discover_candidates
from discogs_recommender.recommend.score import score_candidates
from discogs_recommender.ui.preferences import UserPreferences

logger = logging.getLogger(__name__)

STAGES: list[str] = [
    "build_labels",
    "sync_profile",
    "discover",
    "score",
    "audio_rank",
]

STAGE_ORDER = STAGES


def list_stages() -> list[str]:
    return list(STAGES)


def run_stage(
    name: str,
    config: AppConfig,
    *,
    force: bool = False,
    user_prefs: UserPreferences | None = None,
) -> None:
    if name not in STAGES:
        raise KeyError(f"Unknown stage: {name}. Choose from: {', '.join(STAGES)}")
    logger.info("Running stage: %s", name)
    if name == "build_labels":
        build_label_graph(config, force=force)
    elif name == "sync_profile":
        sync_profile(config, force=force)
    elif name == "discover":
        discover_candidates(config, force=force)
    elif name == "score":
        score_candidates(config, force=force, user_prefs=user_prefs)
    elif name == "audio_rank":
        run_audio_rank(config, force=force)
    logger.info("Finished stage: %s", name)


def run_through(
    last_stage: str,
    config: AppConfig,
    *,
    force: bool = False,
    user_prefs: UserPreferences | None = None,
) -> None:
    if last_stage not in STAGES:
        raise KeyError(f"Unknown stage: {last_stage}")
    idx = STAGES.index(last_stage)
    for name in STAGES[: idx + 1]:
        run_stage(name, config, force=force, user_prefs=user_prefs)
