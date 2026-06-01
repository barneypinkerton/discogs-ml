"""Tests for label family graph logic."""

from discogs_recommender.catalog.label_graph import (
    LabelInfo,
    LabelGraph,
    build_family_graph,
)


def test_build_family_graph_parent_siblings_sublabels():
    label_data = {
        1: LabelInfo(name="Parent", parent_id=None, sublabel_ids=[2, 3]),
        2: LabelInfo(name="Child A", parent_id=1, sublabel_ids=[]),
        3: LabelInfo(name="Child B", parent_id=1, sublabel_ids=[4]),
        4: LabelInfo(name="Grandchild", parent_id=3, sublabel_ids=[]),
    }
    family = build_family_graph(label_data)

    assert 1 in family[2]  # parent
    assert 3 in family[2]  # sibling
    assert 4 in family[3]  # sublabel
    assert 3 in family[4]  # parent of grandchild


def test_label_graph_family_names():
    label_data = {
        1: LabelInfo(name="Parent", parent_id=None, sublabel_ids=[2]),
        2: LabelInfo(name="Child", parent_id=1, sublabel_ids=[]),
    }
    family = build_family_graph(label_data)
    graph = LabelGraph(label_data=label_data, label_family=family)

    names = set(graph.family_names("Child"))
    assert "Parent" in names
