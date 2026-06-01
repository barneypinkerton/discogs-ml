"""Tests for recommendation scoring helpers."""

import pandas as pd

from discogs_recommender.config import AppConfig, DiscoverConfig, DiscogsApiConfig, PathsConfig, ProfileConfig, ScoreConfig
from discogs_recommender.profile.owned import _parse_artist_ids
from discogs_recommender.recommend.score import (
    apply_diversity_cap,
    build_final_recommendations,
    compute_final_score,
    filter_compilations,
)


def _minimal_config() -> AppConfig:
    root = PathsConfig(
        data_root=__import__("pathlib").Path("/tmp"),
        csv_dir=__import__("pathlib").Path("/tmp/csv"),
        db_path=__import__("pathlib").Path("/tmp/db.sqlite"),
        labels_xml=__import__("pathlib").Path("/tmp/labels.xml"),
        catalog_dir=__import__("pathlib").Path("/tmp/catalog"),
        label_data_file=__import__("pathlib").Path("/tmp/label_data.json"),
        label_family_file=__import__("pathlib").Path("/tmp/label_family.json"),
        embeddings_dir=__import__("pathlib").Path("/tmp/embeddings"),
        collection_embeddings=__import__("pathlib").Path("/tmp/c.npz"),
        effnet_model=__import__("pathlib").Path("/tmp/m.pb"),
        exports_dir=__import__("pathlib").Path("/tmp/exports"),
        profile_dir=__import__("pathlib").Path("/tmp/profile"),
        profile_releases=__import__("pathlib").Path("/tmp/profile/releases.json"),
        candidates_file=__import__("pathlib").Path("/tmp/exports/candidates.csv"),
        recommendations_file=__import__("pathlib").Path("/tmp/exports/recs.csv"),
        candidate_audio_dir=__import__("pathlib").Path("/tmp/audio"),
    )
    return AppConfig(
        paths=root,
        discogs=DiscogsApiConfig(user_token="x", username="y"),
        profile=ProfileConfig(),
        discover=DiscoverConfig(),
        score=ScoreConfig(max_per_artist=1, max_per_label=1),
    )


def test_filter_compilations():
    df = pd.DataFrame({"artists": [["Various"], ["Artist A"]]})
    out = filter_compilations(df)
    assert len(out) == 1
    assert out.iloc[0]["artists"] == ["Artist A"]


def test_diversity_cap():
    config = _minimal_config()
    config.score.max_per_artist = 2
    df = pd.DataFrame(
        {
            "artist_ids": ["[1]", "[1]", "[1]", "[2]"],
            "artists": [["A"], ["A"], ["A"], ["B"]],
            "title": ["One", "Two", "Three", "Four"],
            "master_id": [101, 102, 103, 104],
            "labels": [["L1"], ["L2"], ["L3"], ["L4"]],
            "metadata_score": [4.0, 3.0, 2.0, 1.0],
        }
    )
    out = apply_diversity_cap(df, config)
    assert len(out) == 3
    assert (
        sum(_parse_artist_ids(r["artist_ids"])[0] == 1 for _, r in out.iterrows()) == 2
    )


def test_build_final_dedupes_same_master():
    config = _minimal_config()
    config.score.top_n_export = 10
    config.score.max_per_artist = 5
    config.score.max_per_label = 5
    df = pd.DataFrame(
        {
            "release_id": [1, 2, 3],
            "master_id": [999, 999, 1000],
            "artist_ids": ["[1]", "[1]", "[2]"],
            "artists": [["A"], ["A"], ["B"]],
            "labels": [["L1"], ["L1"], ["L2"]],
            "metadata_score": [3.0, 2.0, 1.0],
            "score": [3.0, 2.0, 1.0],
        }
    )
    out = build_final_recommendations(df, config)
    assert list(out["release_id"]) == [1, 3]


def test_build_final_label_cap():
    config = _minimal_config()
    config.score.top_n_export = 10
    config.score.max_per_artist = 10
    config.score.max_per_label = 2
    df = pd.DataFrame(
        {
            "release_id": [1, 2, 3, 4],
            "master_id": [10, 11, 12, 13],
            "label_ids": ["[100]", "[100]", "[100]", "[200]"],
            "artists": [["A"], ["B"], ["C"], ["D"]],
            "labels": [["Same"], ["Same"], ["Same"], ["Other"]],
            "metadata_score": [4.0, 3.0, 2.0, 1.0],
            "score": [4.0, 3.0, 2.0, 1.0],
        }
    )
    out = build_final_recommendations(df, config)
    assert len(out) == 3
    assert sum(1 for _, r in out.iterrows() if r["labels"] == ["Same"]) == 2


def test_compute_final_score_prefers_high_want_low_have():
    config = _minimal_config()
    df = pd.DataFrame(
        {
            "metadata_score": [1.0, 1.0],
            "have_count": [10.0, 10.0],
            "want_count": [5.0, 200.0],
        }
    )
    scored = compute_final_score(df, config)
    # Higher want_count row should rank first after sort
    assert float(scored.iloc[0]["want_count"]) == 200.0
    assert float(scored.iloc[0]["score"]) > float(scored.iloc[1]["score"])
