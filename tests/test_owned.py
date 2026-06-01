"""Tests for owned-release exclusion."""

import pandas as pd

from discogs_recommender.profile.owned import (
    filter_owned_artists,
    filter_owned_releases,
    owned_collection_artist_ids,
    valid_master_id,
)


def test_valid_master_id():
    assert valid_master_id(615361) == 615361
    assert valid_master_id(0) is None
    assert valid_master_id("") is None
    assert valid_master_id(None) is None


def test_filter_owned_releases_by_release_and_master():
    df = pd.DataFrame(
        {
            "release_id": [1, 2, 3, 4],
            "master_id": [100, 200, 200, None],
            "title": ["a", "b", "c", "d"],
        }
    )
    out = filter_owned_releases(df, owned_releases={1}, owned_masters={200})
    assert list(out["release_id"]) == [4]


def test_owned_collection_artist_ids_ignores_wantlist():
    profile = [
        {"source": "collection", "artist_ids": [10, 20]},
        {"source": "wantlist", "artist_ids": [99]},
    ]
    assert owned_collection_artist_ids(profile) == {10, 20}


def test_filter_owned_artists():
    df = pd.DataFrame(
        {
            "release_id": [1, 2, 3],
            "artist_ids": ["[10, 30]", "[40]", "[50]"],
        }
    )
    out = filter_owned_artists(df, {10, 40})
    assert list(out["release_id"]) == [3]
