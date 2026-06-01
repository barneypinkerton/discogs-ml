"""Candidate discovery and scoring."""

from discogs_recommender.recommend.discover import discover_candidates
from discogs_recommender.recommend.score import score_candidates

__all__ = ["discover_candidates", "score_candidates"]
