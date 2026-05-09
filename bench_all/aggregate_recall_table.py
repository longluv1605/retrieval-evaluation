#!/usr/bin/env python3
"""Aggregate recall metrics into CSV and Markdown tables.

Expected layout:
outputs_root/run_id/dataset/scene/map_dir/query_set/method/eval/evaluation.json
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


METHOD_HLOC = "hloc-megaloc"
METHOD_SEQFG = "seqfg"


@dataclass(frozen=True)
class JobKey:
    dataset: str
    scene: str
    map_dir: str
    query_set: str


def _f(v: object) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt(v: float | None) -> str:
    return "NA" if v is None else f"{v:.6f}"


def _collect_eval_paths(run_root: Path) -> List[Path]:
    return sorted(run_root.glob("**/eval/evaluation.json"))


def _parse_eval(path: Path) -> Tuple[JobKey, str, Dict[str, float | None]] | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    method = str(data.get("retrieval_method", ""))
    if method not in {METHOD_HLOC, METHOD_SEQFG}:
        return None

    key = JobKey(
        dataset=str(data.get("dataset", "")),
        scene=str(data.get("scene", "")),
        map_dir=str(data.get("map_dir", "")),
        query_set=str(data.get("query_set", "")),
    )

    recall = data.get("recall", {})
    top1_cdf = data.get("top1_cdf", {})
    dist_stats = data.get("distance_stats_top1", {})
    metrics = {
        "R@1": _f(recall.get("R@1")),
        "R@5": _f(recall.get("R@5")),
        "R@10": _f(recall.get("R@10")),
        "CDF_1m": _f(top1_cdf.get("success_leq_1m")),
        "CDF_2m": _f(top1_cdf.get("success_leq_2m")),
        "CDF_3m": _f(top1_cdf.get("success_leq_3m")),
        "CDF_5m": _f(top1_cdf.get("success_leq_5m")),
        "p90": _f(dist_stats.get("p90")),
        "p95": _f(dist_stats.get("p95")),
    }
    return key, method, metrics


def _winner(delta_r1: float | None) -> str:
    if delta_r1 is None:
        return "unknown"
    if delta_r1 > 0:
        return "seqfg"
    if delta_r1 < 0:
        return "hloc-megaloc"
    return "tie"


def _build_rows(parsed: Iterable[Tuple[JobKey, str, Dict[str, float | None]]], run_id: str) -> List[Dict[str, object]]:
    grouped: Dict[JobKey, Dict[str, Dict[str, float | None]]] = {}
    for key, method, metrics in parsed:
        grouped.setdefault(key, {})[method] = metrics

    rows: List[Dict[str, object]] = []
    for key in sorted(grouped.keys(), key=lambda k: (k.dataset, k.scene, k.map_dir, k.query_set)):
        methods = grouped[key]
        hloc = methods.get(METHOD_HLOC, {})
        seqfg = methods.get(METHOD_SEQFG, {})

        h_r1 = _f(hloc.get("R@1"))
        h_r5 = _f(hloc.get("R@5"))
        h_r10 = _f(hloc.get("R@10"))
        s_r1 = _f(seqfg.get("R@1"))
        s_r5 = _f(seqfg.get("R@5"))
        s_r10 = _f(seqfg.get("R@10"))

        d_r1 = None if h_r1 is None or s_r1 is None else s_r1 - h_r1
        d_r5 = None if h_r5 is None or s_r5 is None else s_r5 - h_r5
        d_r10 = None if h_r10 is None or s_r10 is None else s_r10 - h_r10

        # Top-1 CDF metrics
        cdf_keys = ["CDF_1m", "CDF_2m", "CDF_3m", "CDF_5m"]
        cdf_data: Dict[str, object] = {}
        for ck in cdf_keys:
            h_val = _f(hloc.get(ck))
            s_val = _f(seqfg.get(ck))
            cdf_data[f"hloc_{ck}"] = h_val
            cdf_data[f"seqfg_{ck}"] = s_val
            cdf_data[f"delta_{ck}"] = None if h_val is None or s_val is None else s_val - h_val

        # Distance percentiles
        for pk in ["p90", "p95"]:
            h_val = _f(hloc.get(pk))
            s_val = _f(seqfg.get(pk))
            cdf_data[f"hloc_{pk}"] = h_val
            cdf_data[f"seqfg_{pk}"] = s_val
            cdf_data[f"delta_{pk}"] = None if h_val is None or s_val is None else s_val - h_val

        row: Dict[str, object] = {
            "run_id": run_id,
            "dataset": key.dataset,
            "scene": key.scene,
            "map_dir": key.map_dir,
            "query_set": key.query_set,
            "hloc_R@1": h_r1,
            "seqfg_R@1": s_r1,
            "delta_R@1": d_r1,
            "hloc_R@5": h_r5,
            "seqfg_R@5": s_r5,
            "delta_R@5": d_r5,
            "hloc_R@10": h_r10,
            "seqfg_R@10": s_r10,
            "delta_R@10": d_r10,
            "winner": _winner(d_r1),
        }
        row.update(cdf_data)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "dataset",
        "scene",
        "map_dir",
        "query_set",
        "hloc_R@1",
        "seqfg_R@1",
        "delta_R@1",
        "hloc_R@5",
        "seqfg_R@5",
        "delta_R@5",
        "hloc_R@10",
        "seqfg_R@10",
        "delta_R@10",
        "hloc_CDF_1m",
        "seqfg_CDF_1m",
        "delta_CDF_1m",
        "hloc_CDF_2m",
        "seqfg_CDF_2m",
        "delta_CDF_2m",
        "hloc_CDF_3m",
        "seqfg_CDF_3m",
        "delta_CDF_3m",
        "hloc_CDF_5m",
        "seqfg_CDF_5m",
        "delta_CDF_5m",
        "hloc_p90",
        "seqfg_p90",
        "delta_p90",
        "hloc_p95",
        "seqfg_p95",
        "delta_p95",
        "winner",
    ]
    numeric_fields = [f for f in fields if f.startswith(("hloc_", "seqfg_", "delta_"))]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = dict(r)
            for k in numeric_fields:
                val = out.get(k)
                out[k] = "" if val is None else f"{float(val):.6f}"
            w.writerow(out)


def _write_md(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Recall & CDF Comparison: hloc-megaloc vs seqfg",
        "",
        "## Recall@K",
        "",
        "| dataset | scene | map_dir | query_set | hloc R@1 | seqfg R@1 | ΔR@1 | hloc R@5 | seqfg R@5 | ΔR@5 | hloc R@10 | seqfg R@10 | ΔR@10 | winner |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            "| {dataset} | {scene} | {map_dir} | {query_set} | {h1} | {s1} | {d1} | {h5} | {s5} | {d5} | {h10} | {s10} | {d10} | {winner} |".format(
                dataset=r["dataset"],
                scene=r["scene"],
                map_dir=r["map_dir"],
                query_set=r["query_set"],
                h1=_fmt(_f(r["hloc_R@1"])),
                s1=_fmt(_f(r["seqfg_R@1"])),
                d1=_fmt(_f(r["delta_R@1"])),
                h5=_fmt(_f(r["hloc_R@5"])),
                s5=_fmt(_f(r["seqfg_R@5"])),
                d5=_fmt(_f(r["delta_R@5"])),
                h10=_fmt(_f(r["hloc_R@10"])),
                s10=_fmt(_f(r["seqfg_R@10"])),
                d10=_fmt(_f(r["delta_R@10"])),
                winner=r["winner"],
            )
        )

    lines += [
        "",
        "## Top-1 Distance CDF (Success Rate ≤ τ m)",
        "",
        "| dataset | scene | map_dir | query_set | hloc ≤1m | seqfg ≤1m | Δ≤1m | hloc ≤2m | seqfg ≤2m | Δ≤2m | hloc ≤3m | seqfg ≤3m | Δ≤3m | hloc ≤5m | seqfg ≤5m | Δ≤5m |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        cdf_vals = []
        for tau in ["1m", "2m", "3m", "5m"]:
            h = _f(r.get(f"hloc_CDF_{tau}"))
            s = _f(r.get(f"seqfg_CDF_{tau}"))
            d = _f(r.get(f"delta_CDF_{tau}"))
            cdf_vals.extend([_fmt(h), _fmt(s), _fmt(d)])
        lines.append(
            "| {dataset} | {scene} | {map_dir} | {query_set} | {vals} |".format(
                dataset=r["dataset"],
                scene=r["scene"],
                map_dir=r["map_dir"],
                query_set=r["query_set"],
                vals=" | ".join(cdf_vals),
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate recall tables from evaluation.json files")
    p.add_argument("--outputs-root", type=Path, required=True, help="Outputs root directory")
    p.add_argument("--run-id", type=str, required=True, help="Global run id")
    p.add_argument("--out-csv", type=Path, required=True, help="Output CSV path")
    p.add_argument("--out-md", type=Path, required=True, help="Output Markdown path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.outputs_root / args.run_id
    if not run_root.exists():
        raise SystemExit(f"[ERROR] run root not found: {run_root}")

    eval_paths = _collect_eval_paths(run_root)
    if not eval_paths:
        raise SystemExit(f"[ERROR] no evaluation.json found under: {run_root}")

    parsed = []
    for p in eval_paths:
        row = _parse_eval(p)
        if row is not None:
            parsed.append(row)

    if not parsed:
        raise SystemExit("[ERROR] no hloc-megaloc/seqfg evaluation rows found")

    rows = _build_rows(parsed, args.run_id)
    _write_csv(args.out_csv, rows)
    _write_md(args.out_md, rows)

    print(args.out_csv)
    print(args.out_md)


if __name__ == "__main__":
    main()
