#!/usr/bin/env python3
"""Build benchmark job matrix from generated dataset summary JSON.

Output format (stdout): one job per line
dataset_path|scene|map_split|query_split
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Set, Tuple


Job = Tuple[str, str, str, str]


def _splits(entries: Iterable[dict]) -> List[str]:
    return [e.get("split") for e in entries if e.get("split")]


def _collect_jobs(data: dict) -> List[Job]:
    jobs: List[Job] = []

    for ds in data.get("datasets", []):
        dataset_name = ds.get("dataset")
        if not dataset_name:
            continue

        if dataset_name == "hilti":
            for variant in ds.get("variants", []):
                variant_name = variant.get("variant")
                if not variant_name:
                    continue
                dataset_path = f"hilti/{variant_name}"
                for scene_entry in variant.get("scenes", []):
                    scene = scene_entry.get("scene")
                    if not scene:
                        continue
                    maps = _splits(scene_entry.get("maps", []))
                    queries = _splits(scene_entry.get("subsets", []))
                    for map_split in maps:
                        for query_split in queries:
                            jobs.append((dataset_path, scene, map_split, query_split))
            continue

        dataset_path = dataset_name
        for scene_entry in ds.get("scenes", []):
            scene = scene_entry.get("scene")
            if not scene:
                continue
            maps = _splits(scene_entry.get("maps", []))
            queries = _splits(scene_entry.get("subsets", []))
            for map_split in maps:
                for query_split in queries:
                    jobs.append((dataset_path, scene, map_split, query_split))

    return jobs


def _unique_stable(jobs: Iterable[Job]) -> List[Job]:
    seen: Set[Job] = set()
    out: List[Job] = []
    for job in jobs:
        if job in seen:
            continue
        seen.add(job)
        out.append(job)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build job matrix from summary JSON")
    parser.add_argument("--summary", type=Path, required=True, help="Path to dataset summary JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    jobs = _unique_stable(_collect_jobs(data))
    for job in jobs:
        print("|".join(job))


if __name__ == "__main__":
    main()
