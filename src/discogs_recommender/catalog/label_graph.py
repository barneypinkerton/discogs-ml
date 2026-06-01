"""Label parent / sibling / sublabel family graph.

Ported from archive notebook:
  Versions/v12/code/development/02 DB Set Up/
  02.2_database_setup with label family creation.ipynb
"""

from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from discogs_recommender.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class LabelInfo:
    name: str
    parent_id: int | None
    sublabel_ids: list[int]


@dataclass
class LabelGraph:
    """In-memory label catalog and family adjacency."""

    label_data: dict[int, LabelInfo]
    label_family: dict[int, set[int]]
    _name_to_id: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._name_to_id = {info.name: lid for lid, info in self.label_data.items()}

    def family_ids(self, label_id: int) -> set[int]:
        return self.label_family.get(label_id, set())

    def family_names(self, label_name: str) -> list[str]:
        """Names of related labels (parent, siblings, sublabels)."""
        lid = self._name_to_id.get(label_name)
        if lid is None:
            return []
        return [
            self.label_data[fid].name
            for fid in self.family_ids(lid)
            if fid in self.label_data
        ]

    def resolve_id(self, label_name: str) -> int | None:
        return self._name_to_id.get(label_name)


def _label_info_to_json(lid: int, info: LabelInfo) -> dict[str, Any]:
    return {
        "name": info.name,
        "parent_id": info.parent_id,
        "sublabel_ids": info.sublabel_ids,
    }


def _label_info_from_json(raw: dict[str, Any]) -> LabelInfo:
    return LabelInfo(
        name=raw["name"],
        parent_id=raw.get("parent_id"),
        sublabel_ids=list(raw.get("sublabel_ids", [])),
    )


def save_label_graph(
    label_data: dict[int, LabelInfo],
    label_family: dict[int, set[int]],
    label_data_path: Path,
    label_family_path: Path,
) -> None:
    label_data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_data_path, "w", encoding="utf-8") as f:
        json.dump(
            {str(k): _label_info_to_json(k, v) for k, v in label_data.items()},
            f,
        )
    with open(label_family_path, "w", encoding="utf-8") as f:
        json.dump(
            {str(k): list(v) for k, v in label_family.items()},
            f,
        )
    logger.info("Saved %s and %s", label_data_path, label_family_path)


def load_label_graph_from_json(
    label_data_path: Path,
    label_family_path: Path,
) -> tuple[dict[int, LabelInfo], dict[int, set[int]]]:
    with open(label_data_path, encoding="utf-8") as f:
        raw_data = json.load(f)
    with open(label_family_path, encoding="utf-8") as f:
        raw_family = json.load(f)

    label_data = {int(k): _label_info_from_json(v) for k, v in raw_data.items()}
    label_family = {int(k): set(v) for k, v in raw_family.items()}
    return label_data, label_family


def parse_labels_xml(
    labels_xml: Path,
    *,
    progress_every: int = 500_000,
) -> dict[int, LabelInfo]:
    """Parse Discogs labels XML dump into label records."""
    if not labels_xml.is_file():
        raise FileNotFoundError(f"Labels XML not found: {labels_xml}")

    label_data: dict[int, LabelInfo] = {}
    start = time.time()
    count = 0

    logger.info("Parsing labels from %s", labels_xml)
    for event, elem in ET.iterparse(str(labels_xml), events=("end",)):
        if elem.tag == "label" and elem.find("id") is not None:
            label_id = int(elem.findtext("id"))
            name = elem.findtext("name") or ""

            parent_el = elem.find("parentLabel")
            parent_id = None
            if parent_el is not None and parent_el.attrib.get("id"):
                parent_id = int(parent_el.attrib["id"])

            sublabel_ids: list[int] = []
            sublabels_el = elem.find("sublabels")
            if sublabels_el is not None:
                for sub in sublabels_el.findall("label"):
                    sid = sub.attrib.get("id")
                    if sid:
                        sublabel_ids.append(int(sid))

            label_data[label_id] = LabelInfo(
                name=name,
                parent_id=parent_id,
                sublabel_ids=sublabel_ids,
            )
            count += 1
            if progress_every and count % progress_every == 0:
                logger.info("Parsed %s labels...", count)
            elem.clear()

    logger.info("Parsed %s labels in %.1fs", len(label_data), time.time() - start)
    return label_data


def build_family_graph(label_data: dict[int, LabelInfo]) -> dict[int, set[int]]:
    """Build parent + sibling + sublabel adjacency for each label."""
    label_family: dict[int, set[int]] = {}
    for lid, info in label_data.items():
        family: set[int] = set()

        if info.parent_id is not None:
            family.add(info.parent_id)
            parent_info = label_data.get(info.parent_id)
            if parent_info:
                for sib_id in parent_info.sublabel_ids:
                    if sib_id != lid:
                        family.add(sib_id)

        for sub_id in info.sublabel_ids:
            family.add(sub_id)

        if family:
            label_family[lid] = family

    logger.info(
        "Built family graph: %s labels with connections", len(label_family)
    )
    return label_family


def load_label_graph(config: AppConfig) -> LabelGraph:
    """Load label graph from JSON cache files."""
    paths = config.paths
    label_data, label_family = load_label_graph_from_json(
        paths.label_data_file,
        paths.label_family_file,
    )
    return LabelGraph(label_data=label_data, label_family=label_family)


def build_label_graph(config: AppConfig, *, force: bool = False) -> LabelGraph | None:
    """
    Ensure label graph JSON exists, or parse labels XML and write JSON.

    When cache files already exist, this stage only verifies their presence
    (it does not load multi-GB JSON into memory). Use :func:`load_label_graph`
    when you need an in-memory :class:`LabelGraph`.
    """
    paths = config.paths
    data_path = paths.label_data_file
    family_path = paths.label_family_file

    if not force and data_path.is_file() and family_path.is_file():
        logger.info(
            "Label graph cache present (%s MB + %s MB); skipping parse. "
            "Use load_label_graph() when you need in-memory access.",
            data_path.stat().st_size // (1024 * 1024),
            family_path.stat().st_size // (1024 * 1024),
        )
        return None

    label_data = parse_labels_xml(
        paths.labels_xml,
        progress_every=config.label_graph_parse_progress_every,
    )
    label_family = build_family_graph(label_data)
    save_label_graph(label_data, label_family, data_path, family_path)
    return LabelGraph(label_data=label_data, label_family=label_family)
