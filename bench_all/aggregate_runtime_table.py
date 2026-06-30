#!/usr/bin/env python3
"""Aggregate retrieval runtime logs into CSV and Markdown tables.

Expected layout:
outputs_root/run_id/dataset/scene/map_dir/query_set/method/meta/retrieval_runtime.json
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any


METHODS = {"hloc-megaloc", "seqfg"}


@dataclass(frozen=True)
class RuntimeRow:
    run_id: str
    dataset: str
    scene: str
    map_dir: str
    query_set: str
    method: str
    num_map: int
    num_query: int
    valid_for_no_extraction_benchmark: bool
    features_precomputed: bool
    feature_extraction_s: float
    descriptor_load_s: float
    similarity_search_s: float
    ranking_s: float
    retrieval_core_s: float
    retrieval_without_extraction_s: float
    retrieval_core_ms_per_query: float
    retrieval_without_extraction_ms_per_query: float
    pairs_write_s: float
    retrieve_command_wall_s: float
    path: str


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_runtime_path(path: Path, run_root: Path) -> tuple[str, str, str, str, str]:
    rel = path.relative_to(run_root)
    parts = rel.parts
    if len(parts) < 7 or parts[-2:] != ("meta", "retrieval_runtime.json"):
        raise ValueError(f"Unexpected runtime path layout: {path}")
    dataset, scene, map_dir, query_set, method = parts[:5]
    return dataset, scene, map_dir, query_set, method


def _parse_runtime(path: Path, run_root: Path, run_id: str) -> RuntimeRow | None:
    dataset, scene, map_dir, query_set, method_from_path = _parse_runtime_path(path, run_root)
    data = json.loads(path.read_text(encoding="utf-8"))
    method = str(data.get("method", method_from_path))
    if method not in METHODS:
        return None

    runtime = data.get("pair_selection_runtime", {})
    if not isinstance(runtime, dict):
        runtime = {}

    return RuntimeRow(
        run_id=run_id,
        dataset=dataset,
        scene=scene,
        map_dir=map_dir,
        query_set=query_set,
        method=method,
        num_map=_i(runtime.get("num_map"), _i(data.get("num_map"))),
        num_query=_i(runtime.get("num_query"), _i(data.get("num_query"))),
        valid_for_no_extraction_benchmark=bool(runtime.get("valid_for_no_extraction_benchmark", False)),
        features_precomputed=bool(runtime.get("features_precomputed", False)),
        feature_extraction_s=_f(runtime.get("feature_extraction_s")),
        descriptor_load_s=_f(runtime.get("descriptor_load_s")),
        similarity_search_s=_f(runtime.get("similarity_search_s")),
        ranking_s=_f(runtime.get("ranking_s")),
        retrieval_core_s=_f(runtime.get("retrieval_core_s")),
        retrieval_without_extraction_s=_f(runtime.get("retrieval_without_extraction_s")),
        retrieval_core_ms_per_query=_f(runtime.get("retrieval_core_ms_per_query")),
        retrieval_without_extraction_ms_per_query=_f(runtime.get("retrieval_without_extraction_ms_per_query")),
        pairs_write_s=_f(data.get("pairs_write_s")),
        retrieve_command_wall_s=_f(data.get("retrieve_command_wall_s")),
        path=str(path),
    )


def _collect_rows(outputs_root: Path, run_id: str) -> list[RuntimeRow]:
    run_root = outputs_root / run_id
    if not run_root.exists():
        raise SystemExit(f"[ERROR] run root not found: {run_root}")

    paths = sorted(run_root.glob("**/meta/retrieval_runtime.json"))
    if not paths:
        raise SystemExit(f"[ERROR] no retrieval_runtime.json found under: {run_root}")

    rows: list[RuntimeRow] = []
    for path in paths:
        row = _parse_runtime(path, run_root, run_id)
        if row is not None:
            rows.append(row)
    if not rows:
        raise SystemExit("[ERROR] no hloc-megaloc/seqfg runtime rows found")
    return rows


def _write_csv(path: Path, rows: list[RuntimeRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(RuntimeRow.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _fmt(v: float) -> str:
    return f"{v:.6f}"


def _write_md(path: Path, rows: list[RuntimeRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Retrieval Runtime",
        "",
        "Scope: descriptor extraction is excluded from `retrieval_without_extraction_s`.",
        "",
        "| dataset | scene | map_dir | query_set | method | valid | queries | core ms/query | no-extract ms/query | no-extract s |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {scene} | {map_dir} | {query_set} | {method} | {valid} | {queries} | {core_ms} | {noext_ms} | {noext_s} |".format(
                dataset=row.dataset,
                scene=row.scene,
                map_dir=row.map_dir,
                query_set=row.query_set,
                method=row.method,
                valid="yes" if row.valid_for_no_extraction_benchmark else "no",
                queries=row.num_query,
                core_ms=_fmt(row.retrieval_core_ms_per_query),
                noext_ms=_fmt(row.retrieval_without_extraction_ms_per_query),
                noext_s=_fmt(row.retrieval_without_extraction_s),
            )
        )

    valid_rows = [r for r in rows if r.valid_for_no_extraction_benchmark]
    if valid_rows:
        lines += [
            "",
            "## Summary By Method",
            "",
            "| method | runs | total queries | median no-extract ms/query | mean no-extract ms/query | median core ms/query | mean core ms/query |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for method in sorted({r.method for r in valid_rows}):
            subset = [r for r in valid_rows if r.method == method]
            noext = [r.retrieval_without_extraction_ms_per_query for r in subset]
            core = [r.retrieval_core_ms_per_query for r in subset]
            lines.append(
                "| {method} | {runs} | {queries} | {med_noext} | {mean_noext} | {med_core} | {mean_core} |".format(
                    method=method,
                    runs=len(subset),
                    queries=sum(r.num_query for r in subset),
                    med_noext=_fmt(median(noext)),
                    mean_noext=_fmt(mean(noext)),
                    med_core=_fmt(median(core)),
                    mean_core=_fmt(mean(core)),
                )
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate retrieval runtime logs")
    parser.add_argument("--outputs-root", type=Path, required=True, help="Outputs root directory")
    parser.add_argument("--run-id", type=str, required=True, help="Global run id")
    parser.add_argument("--out-csv", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--out-md", type=Path, required=True, help="Output Markdown path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _collect_rows(args.outputs_root, args.run_id)
    _write_csv(args.out_csv, rows)
    _write_md(args.out_md, rows)
    print(args.out_csv)
    print(args.out_md)


if __name__ == "__main__":
    main()
