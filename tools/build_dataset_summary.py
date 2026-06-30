#!/usr/bin/env python3
"""Generate grouped dataset summary from a dataset root.

Minimal behavior:
- Auto-discover datasets/scenes/maps/subsets from filesystem.
- Keep Hilti variant level (single/equirect/etc.) if present.
- Compute n_poses, bbox area, xyz span, and path length from poses.txt.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def count_lines(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def read_xyz_points(pose_file: Path) -> List[Tuple[float, float, float]]:
    if not pose_file.exists():
        return []
    points: List[Tuple[float, float, float]] = []
    with pose_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                points.append((float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError:
                continue
    return points


def summarize_pose_file(pose_file: Path) -> Dict[str, Any]:
    pts = read_xyz_points(pose_file)
    if not pts:
        return {
            "n_poses": None,
            "bbox_area_xy_m2": None,
            "span_m": None,
            "path_length_m": None,
        }

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    span_z = max(zs) - min(zs)

    path_len = 0.0
    if len(pts) > 1:
        prev = pts[0]
        for p in pts[1:]:
            path_len += math.dist(prev, p)
            prev = p

    return {
        "n_poses": len(pts),
        "bbox_area_xy_m2": round(span_x * span_y, 3),
        "span_m": [round(span_x, 3), round(span_y, 3), round(span_z, 3)],
        "path_length_m": round(path_len, 3),
    }


def classify_scene_dirs(scene_root: Path) -> Tuple[List[Path], List[Path]]:
    maps: List[Path] = []
    subsets: List[Path] = []
    for d in sorted([p for p in scene_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        n = d.name
        if n == "map" or n.endswith("_map"):
            maps.append(d)
        elif n in {"query", "query_cam1"} or n.startswith("query_") or n.endswith("_query"):
            subsets.append(d)
    return maps, subsets


def infer_status(map_entries: List[Dict[str, Any]]) -> str:
    return "READY" if any(m.get("n_poses") is not None for m in map_entries) else "PARTIAL"


def map_entry(map_dir: Path) -> Dict[str, Any]:
    s = summarize_pose_file(map_dir / "poses.txt")
    return {
        "split": map_dir.name,
        "n_poses": s["n_poses"],
        "bbox_area_xy_m2": s["bbox_area_xy_m2"],
        "span_m": s["span_m"],
        "path_length_m": s["path_length_m"],
    }


def subset_entry(subset_dir: Path) -> Dict[str, Any]:
    s = summarize_pose_file(subset_dir / "poses.txt")
    return {
        "split": subset_dir.name,
        "n_poses": s["n_poses"],
        "query_names": count_lines(subset_dir / "query_names.txt"),
        "path_length_m": s["path_length_m"],
    }


def build_summary(dataset_root: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "generated_at": str(date.today()),
        "root": str(dataset_root),
        "datasets": [],
    }

    for dataset_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        dataset = dataset_dir.name

        if dataset == "hilti":
            variants = []
            for variant_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
                scenes = []
                for scene_dir in sorted([p for p in variant_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
                    map_dirs, subset_dirs = classify_scene_dirs(scene_dir)
                    maps = [map_entry(m) for m in map_dirs]
                    subsets = [subset_entry(s) for s in subset_dirs]
                    scenes.append(
                        {
                            "scene": scene_dir.name,
                            "status": infer_status(maps),
                            "maps": maps,
                            "subsets": subsets,
                        }
                    )
                variants.append({"variant": variant_dir.name, "scenes": scenes})
            out["datasets"].append({"dataset": dataset, "variants": variants})
            continue

        scenes = []
        for scene_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            map_dirs, subset_dirs = classify_scene_dirs(scene_dir)
            maps = [map_entry(m) for m in map_dirs]
            subsets = [subset_entry(s) for s in subset_dirs]
            scenes.append(
                {
                    "scene": scene_dir.name,
                    "status": infer_status(maps),
                    "maps": maps,
                    "subsets": subsets,
                }
            )
        out["datasets"].append({"dataset": dataset, "scenes": scenes})

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build grouped dataset summary JSON.")
    p.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to datasets root (required).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/home/long/Workspace/thesis/experiments/resources/benchmarks/dataset_summary.generated.json"),
        help="Output summary JSON.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset_root.exists() or not args.dataset_root.is_dir():
        raise SystemExit(f"[ERROR] dataset root not found: {args.dataset_root}")

    summary = build_summary(args.dataset_root)
    save_json(args.output, summary)
    print(args.output)


if __name__ == "__main__":
    main()
