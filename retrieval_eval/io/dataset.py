from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetLayout:
    dataset_root: Path
    dataset: str
    scene: str
    query_set: str
    map_dir: str = "map"

    @property
    def scene_root(self) -> Path:
        return self.dataset_root / self.dataset / self.scene

    @property
    def map_root(self) -> Path:
        return self.scene_root / self.map_dir

    @property
    def query_root(self) -> Path:
        return self.scene_root / self.query_set

    @property
    def map_images_dir(self) -> Path:
        return self.map_root / "images"

    @property
    def query_images_dir(self) -> Path:
        return self.query_root / "images"


def read_image_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing image list: {path}")
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    names = [ln for ln in lines if ln and not ln.startswith("#")]
    if not names:
        raise ValueError(f"Empty image list: {path}")
    return names


def get_map_names(layout: DatasetLayout) -> list[str]:
    return read_image_list(layout.map_root / "images.txt")


def get_query_names(layout: DatasetLayout) -> list[str]:
    query_names = layout.query_root / "query_names.txt"
    if query_names.exists():
        names = read_image_list(query_names)
        sequence_file = layout.query_root / "query_sequences.json"
        if sequence_file.exists():
            payload = json.loads(sequence_file.read_text(encoding="utf-8"))
            if isinstance(payload, list) and all(isinstance(group, list) for group in payload):
                annotated = [
                    name
                    for group in payload
                    for name in group
                    if isinstance(name, str)
                ]
                annotated_set = set(annotated)
                looks_truncated = (
                    len(annotated) == len(annotated_set)
                    and len(annotated) >= 2 * len(names)
                    and set(names).issubset(annotated_set)
                )
                if looks_truncated:
                    return annotated
        return names
    return read_image_list(layout.query_root / "images.txt")
