#!/usr/bin/env python3
"""Export full-resolution retrieval results into a browsable query gallery."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps


@dataclass(frozen=True, order=True)
class RunKey:
    dataset: str
    scene: str
    map_dir: str
    query_set: str


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_evaluations(
    outputs_root: Path,
    run_id: str,
    method: str,
    datasets: set[str],
) -> dict[RunKey, Path]:
    found: dict[RunKey, Path] = {}
    for path in (outputs_root / run_id).glob("**/eval/evaluation.json"):
        payload = read_json(path)
        if str(payload.get("retrieval_method")) != method:
            continue
        dataset = str(payload.get("dataset"))
        if datasets and dataset not in datasets:
            continue
        key = RunKey(
            dataset=dataset,
            scene=str(payload.get("scene")),
            map_dir=str(payload.get("map_dir")),
            query_set=str(payload.get("query_set")),
        )
        found[key] = path
    return found


def read_scored_pairs(path: Path, top_k: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            rank = int(raw["rank"])
            if rank > top_k:
                continue
            grouped[str(raw["query_name"])].append(
                {
                    "map_name": str(raw["map_name"]),
                    "score": float(raw["score"]),
                    "rank": rank,
                    "distance_m": float(raw["distance_m"]),
                    "is_positive": bool(int(raw["is_positive"])),
                }
            )
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["rank"]))
    return dict(grouped)


def safe_component(value: str, max_len: int = 80) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    cleaned = cleaned.strip("._") or "item"
    if len(cleaned) <= max_len:
        return cleaned
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned[: max_len - 11]}_{digest}"


def resolve_image(images_root: Path, name: str) -> Path:
    path = images_root / name
    if path.is_file():
        return path
    raise FileNotFoundError(f"Image not found: {path}")


def materialize(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        try:
            os.link(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown materialization mode: {mode}")


def exported_name(row: dict[str, Any], source: Path) -> str:
    status = "hit" if row["is_positive"] else "miss"
    distance = f"{float(row['distance_m']):07.2f}m".replace(".", "p")
    return (
        f"top_{int(row['rank']):02d}__{status}__d_{distance}__"
        f"{safe_component(source.stem, 48)}{source.suffix.lower()}"
    )


def load_panel(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        fitted = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def draw_preview(
    path: Path,
    query_path: Path,
    baseline_path: Path,
    seqfg_path: Path,
    baseline_row: dict[str, Any],
    seqfg_row: dict[str, Any],
    title: str,
) -> None:
    panel_size = (480, 320)
    header_h = 70
    label_h = 54
    gap = 12
    width = panel_size[0] * 3 + gap * 4
    height = header_h + panel_size[1] + label_h + gap * 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=18)
    draw.text((gap, 16), title, fill="#111111", font=font)

    panels = [
        ("Query", query_path, None),
        ("Independent top-1", baseline_path, baseline_row),
        ("SeqFG-Time top-1", seqfg_path, seqfg_row),
    ]
    for index, (label, image_path, row) in enumerate(panels):
        x = gap + index * (panel_size[0] + gap)
        y = header_h
        canvas.paste(load_panel(image_path, panel_size), (x, y))
        border = "#555555"
        if row is not None:
            border = "#18864b" if row["is_positive"] else "#c4372d"
        draw.rectangle(
            (x - 2, y - 2, x + panel_size[0] + 1, y + panel_size[1] + 1),
            outline=border,
            width=4,
        )
        detail = label
        if row is not None:
            status = "hit" if row["is_positive"] else "miss"
            detail += f" | {float(row['distance_m']):.2f} m | {status}"
        draw.text((x, y + panel_size[1] + 12), detail, fill=border, font=small)

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, "JPEG", quality=88, optimize=True)


def category_for(baseline_top1: dict[str, Any], seqfg_top1: dict[str, Any]) -> tuple[str, str]:
    baseline_hit = bool(baseline_top1["is_positive"])
    seqfg_hit = bool(seqfg_top1["is_positive"])
    if seqfg_hit:
        return "seqfg_success", "promoted" if not baseline_hit else "retained_success"
    return "seqfg_fail", "regressed" if baseline_hit else "persistent_failure"


def iter_common_queries(
    baseline: dict[str, list[dict[str, Any]]],
    seqfg: dict[str, list[dict[str, Any]]],
) -> Iterable[str]:
    return sorted(
        query
        for query in set(baseline) & set(seqfg)
        if baseline[query] and seqfg[query]
    )


def export_query(
    *,
    out_root: Path,
    dataset_root: Path,
    key: RunKey,
    query_name: str,
    baseline_rows: list[dict[str, Any]],
    seqfg_rows: list[dict[str, Any]],
    materialize_mode: str,
    make_preview: bool,
) -> dict[str, Any]:
    top_category, detail_category = category_for(baseline_rows[0], seqfg_rows[0])
    query_hash = hashlib.sha1(query_name.encode("utf-8")).hexdigest()[:10]
    query_id = f"{safe_component(Path(query_name).stem, 54)}_{query_hash}"
    config_id = "__".join(
        safe_component(value, 36)
        for value in (key.dataset, key.scene, key.map_dir, key.query_set)
    )
    query_dir = out_root / top_category / detail_category / config_id / query_id

    scene_root = dataset_root / key.dataset / key.scene
    query_source = resolve_image(scene_root / key.query_set / "images", query_name)
    query_export = query_dir / "query" / f"query{query_source.suffix.lower()}"
    materialize(query_source, query_export, materialize_mode)

    exported_methods: dict[str, list[dict[str, Any]]] = {}
    for method_dir, rows in (("baseline", baseline_rows), ("seqfg", seqfg_rows)):
        exported_rows: list[dict[str, Any]] = []
        for row in rows:
            source = resolve_image(scene_root / key.map_dir / "images", str(row["map_name"]))
            destination = query_dir / method_dir / exported_name(row, source)
            materialize(source, destination, materialize_mode)
            exported_rows.append(
                {
                    **row,
                    "source_path": str(source.resolve()),
                    "exported_path": str(destination.relative_to(out_root)),
                }
            )
        exported_methods[method_dir] = exported_rows

    preview_path = query_dir / "preview.jpg"
    if make_preview:
        baseline_top_path = out_root / exported_methods["baseline"][0]["exported_path"]
        seqfg_top_path = out_root / exported_methods["seqfg"][0]["exported_path"]
        draw_preview(
            preview_path,
            query_export,
            baseline_top_path,
            seqfg_top_path,
            baseline_rows[0],
            seqfg_rows[0],
            title=f"{key.dataset}/{key.scene}/{key.map_dir}/{key.query_set}",
        )

    metadata = {
        "category": top_category,
        "detail_category": detail_category,
        "dataset": key.dataset,
        "scene": key.scene,
        "map_dir": key.map_dir,
        "query_set": key.query_set,
        "query_name": query_name,
        "query_source_path": str(query_source.resolve()),
        "query_exported_path": str(query_export.relative_to(out_root)),
        "preview_path": str(preview_path.relative_to(out_root)) if make_preview else None,
        "baseline": exported_methods["baseline"],
        "seqfg": exported_methods["seqfg"],
    }
    (query_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return {
        "pair_id": f"{config_id}__{query_id}",
        "category": top_category,
        "detail_category": detail_category,
        "dataset": key.dataset,
        "scene": key.scene,
        "map_dir": key.map_dir,
        "query_set": key.query_set,
        "query_name": query_name,
        "baseline_top1_name": baseline_rows[0]["map_name"],
        "baseline_top1_distance_m": baseline_rows[0]["distance_m"],
        "baseline_top1_hit": int(bool(baseline_rows[0]["is_positive"])),
        "seqfg_top1_name": seqfg_rows[0]["map_name"],
        "seqfg_top1_distance_m": seqfg_rows[0]["distance_m"],
        "seqfg_top1_hit": int(bool(seqfg_rows[0]["is_positive"])),
        "query_dir": str(query_dir.relative_to(out_root)),
        "preview_path": str(preview_path.relative_to(out_root)) if make_preview else "",
    }


def write_index(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path) -> None:
    path.write_text(
        """# Retrieval gallery

Each query directory contains:

- `preview.jpg`: query, independent top-1, and SeqFG-Time top-1.
- `query/`: the full-resolution query image.
- `baseline/`: full-resolution independent retrievals, named by rank/status/distance.
- `seqfg/`: full-resolution SeqFG retrievals, named by rank/status/distance.
- `metadata.json`: exact source paths, scores, ranks, distances, and hit labels.

Top-level categories:

- `seqfg_success/promoted`: baseline top-1 misses and SeqFG top-1 hits.
- `seqfg_success/retained_success`: both methods hit.
- `seqfg_fail/regressed`: baseline top-1 hits and SeqFG top-1 misses.
- `seqfg_fail/persistent_failure`: both methods miss.

Use `index.csv` to filter and sort cases. To request a paper figure, provide the
`pair_id` values from that file. The exported full-resolution images are
hardlinks by default, so do not edit them in place; edits would affect the
dataset source image.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export query images and full-resolution baseline/SeqFG retrievals, "
            "grouped by SeqFG top-1 success or failure."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--baseline-run-id", required=True)
    parser.add_argument("--seqfg-run-id", required=True)
    parser.add_argument("--baseline-method", default="hloc-megaloc")
    parser.add_argument("--seqfg-method", default="seqfg")
    parser.add_argument("--datasets", nargs="*", default=["croco", "lamar"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--materialize",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How to expose full-resolution source images (default: hardlink).",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip the query/baseline/SeqFG top-1 JPEG preview.",
    )
    parser.add_argument(
        "--limit-per-config",
        type=int,
        default=0,
        help="Debug limit; zero exports every common query.",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        raise SystemExit("--top-k must be >= 1")

    datasets = set(args.datasets)
    baseline_evals = find_evaluations(
        args.outputs_root, args.baseline_run_id, args.baseline_method, datasets
    )
    seqfg_evals = find_evaluations(
        args.outputs_root, args.seqfg_run_id, args.seqfg_method, datasets
    )
    keys = sorted(set(baseline_evals) & set(seqfg_evals))
    if not keys:
        raise SystemExit("No common dataset/scene/map/query configurations found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    missing_images: list[str] = []

    for key in keys:
        baseline_eval = read_json(baseline_evals[key])
        seqfg_eval = read_json(seqfg_evals[key])
        if float(baseline_eval["distance_threshold_m"]) != float(seqfg_eval["distance_threshold_m"]):
            raise RuntimeError(f"Threshold mismatch for {key}")
        baseline_pairs = read_scored_pairs(
            baseline_evals[key].parent / "pairs_scored.csv", args.top_k
        )
        seqfg_pairs = read_scored_pairs(
            seqfg_evals[key].parent / "pairs_scored.csv", args.top_k
        )
        queries = list(iter_common_queries(baseline_pairs, seqfg_pairs))
        if args.limit_per_config:
            queries = queries[: args.limit_per_config]

        for query_name in queries:
            try:
                row = export_query(
                    out_root=args.out_dir,
                    dataset_root=args.dataset_root,
                    key=key,
                    query_name=query_name,
                    baseline_rows=baseline_pairs[query_name],
                    seqfg_rows=seqfg_pairs[query_name],
                    materialize_mode=args.materialize,
                    make_preview=not args.no_preview,
                )
            except (FileNotFoundError, OSError) as exc:
                missing_images.append(f"{key}: {query_name}: {exc}")
                continue
            index_rows.append(row)
            counts[row["category"]] += 1
            counts[row["detail_category"]] += 1

        print(f"[done] {key}: {len(queries)} queries")

    index_rows.sort(
        key=lambda row: (
            row["category"],
            row["detail_category"],
            row["dataset"],
            row["scene"],
            row["map_dir"],
            row["query_set"],
            row["query_name"],
        )
    )
    write_index(args.out_dir / "index.csv", index_rows)
    write_readme(args.out_dir / "README.md")
    summary = {
        "dataset_root": str(args.dataset_root.resolve()),
        "outputs_root": str(args.outputs_root.resolve()),
        "baseline_run_id": args.baseline_run_id,
        "seqfg_run_id": args.seqfg_run_id,
        "baseline_method": args.baseline_method,
        "seqfg_method": args.seqfg_method,
        "datasets": sorted(datasets),
        "top_k": args.top_k,
        "materialize": args.materialize,
        "previews": not args.no_preview,
        "num_configurations": len(keys),
        "num_exported_queries": len(index_rows),
        "counts": dict(sorted(counts.items())),
        "num_missing_images": len(missing_images),
        "missing_images": missing_images,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
