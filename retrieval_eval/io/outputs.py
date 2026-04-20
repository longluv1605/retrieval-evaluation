from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OutputLayout:
    outputs_root: Path
    dataset: str
    scene: str
    map_dir: str
    query_set: str
    retrieval_method: str
    run_id: str

    @property
    def run_root(self) -> Path:
        return (
            self.outputs_root
            / self.run_id
            / self.dataset
            / self.scene
            / self.map_dir
            / self.query_set
            / self.retrieval_method
        )

    @property
    def retrieval_dir(self) -> Path:
        return self.run_root / "retrieval"

    @property
    def eval_dir(self) -> Path:
        return self.run_root / "eval"

    @property
    def meta_dir(self) -> Path:
        return self.run_root / "meta"


def ensure_dirs(layout: OutputLayout) -> None:
    for d in [layout.retrieval_dir, layout.eval_dir, layout.meta_dir]:
        d.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def write_stage_status(path: Path, stage: str, status: str, details: dict[str, Any]) -> None:
    payload = {
        "stage": stage,
        "status": status,
        "details": details,
    }
    write_json(path, payload)
