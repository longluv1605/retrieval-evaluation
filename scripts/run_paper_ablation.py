#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval_eval.config import load_yaml
from retrieval_eval.run import run


PAPER_JOBS = [
    ("croco", "ARCHE_D2", "hl_map", "hl_query"),
    ("croco", "ARCHE_D2", "hl_map", "ios_query"),
    ("croco", "ARCHE_D2", "hl_map", "spot_query"),
    ("croco", "ARCHE_D2", "ios_map", "hl_query"),
    ("croco", "ARCHE_D2", "ios_map", "ios_query"),
    ("croco", "ARCHE_D2", "ios_map", "spot_query"),
    ("croco", "ARCHE_D2", "spot_map", "hl_query"),
    ("croco", "ARCHE_D2", "spot_map", "ios_query"),
    ("croco", "ARCHE_D2", "spot_map", "spot_query"),
    ("croco", "HYDRO", "hl_map", "hl_query"),
    ("croco", "HYDRO", "hl_map", "ios_query"),
    ("lamar", "HGE", "map", "query_val_hololens"),
    ("lamar", "HGE", "map", "query_val_phone"),
]


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_trials(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Ablation trials must be a non-empty JSON list.")
    return payload


def _read_per_query(eval_path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in (eval_path.parent / "per_query.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("evaluated"):
                rows.append(row)
    return rows


def _aggregate(eval_paths: list[Path]) -> dict[str, Any]:
    rows = [row for path in eval_paths for row in _read_per_query(path)]
    distances = np.asarray([float(row["top1_distance_m"]) for row in rows], dtype=np.float64)
    n = len(rows)
    hits = {
        k: sum(bool(row["hit_at"][str(k)]) for row in rows)
        for k in (1, 5, 10, 20, 50, 100)
        if str(k) in rows[0]["hit_at"]
    }

    runtime_total = 0.0
    runtime_core = 0.0
    for path in eval_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        runtime = payload["retrieval_config"]["params"].get("runtime", {})
        runtime_total += float(runtime.get("retrieval_without_extraction_s", 0.0))
        runtime_core += float(runtime.get("retrieval_core_s", 0.0))

    return {
        "num_queries": n,
        "recall": {f"R@{k}": hits[k] / n for k in hits},
        "distance_stats_top1": {
            "mean": float(np.mean(distances)),
            "median": float(np.median(distances)),
            "p90": float(np.percentile(distances, 90)),
        },
        "runtime": {
            "retrieval_without_extraction_s": runtime_total,
            "retrieval_core_s": runtime_core,
            "retrieval_without_extraction_ms_per_query": 1000.0 * runtime_total / n,
            "retrieval_core_ms_per_query": 1000.0 * runtime_core / n,
        },
    }


def _run_matrix(
    *,
    dataset_root: Path,
    outputs_root: Path,
    run_id: str,
    method: str,
    retrieval_config: dict[str, Any],
    eval_config: dict[str, Any],
    threshold_m: float,
) -> list[Path]:
    eval_paths = []
    for dataset, scene, map_dir, query_set in PAPER_JOBS:
        print(f"[{run_id}] {dataset}/{scene}/{map_dir}/{query_set}")
        _, eval_path = run(
            dataset_root=dataset_root,
            outputs_root=outputs_root,
            dataset_name=dataset,
            scene=scene,
            map_dir=map_dir,
            query_set=query_set,
            retrieval_method=method,
            run_id=run_id,
            retrieval_config=retrieval_config,
            eval_config=eval_config,
            threshold_m=threshold_m,
        )
        eval_paths.append(eval_path)
    return eval_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ICARCV paper ablation matrix.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=REPO_ROOT / "configs/retrieval.sequence.legacy_timestamp.yaml",
    )
    parser.add_argument(
        "--trials",
        type=Path,
        default=REPO_ROOT / "configs/paper_ablation_trials.json",
    )
    parser.add_argument(
        "--hloc-top100-config",
        type=Path,
        default=REPO_ROOT / "configs/retrieval.hloc.top100.yaml",
    )
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=REPO_ROOT / "configs/eval.yaml",
    )
    parser.add_argument(
        "--eval-top100-config",
        type=Path,
        default=REPO_ROOT / "configs/eval.top100.yaml",
    )
    parser.add_argument("--threshold-m", type=float, default=5.0)
    parser.add_argument("--tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--skip-oracle", action="store_true")
    args = parser.parse_args()

    base_cfg = load_yaml(args.base_config)
    eval_cfg = load_yaml(args.eval_config)
    eval_top100_cfg = load_yaml(args.eval_top100_config)
    trials = _read_trials(args.trials)
    report: dict[str, Any] = {
        "tag": args.tag,
        "threshold_m": args.threshold_m,
        "jobs": ["/".join(job) for job in PAPER_JOBS],
        "base_config": str(args.base_config),
        "trials": {},
    }

    for trial in trials:
        name = str(trial["name"])
        patch = {key: value for key, value in trial.items() if key != "name"}
        cfg = _deep_merge(base_cfg, {"retrieval": patch})
        cfg["_config_path"] = f"{args.base_config} + trial:{name}"
        run_id = f"paper_ablation_{args.tag}_{name}"
        paths = _run_matrix(
            dataset_root=args.dataset_root,
            outputs_root=args.outputs_root,
            run_id=run_id,
            method="seqfg",
            retrieval_config=cfg,
            eval_config=eval_cfg,
            threshold_m=args.threshold_m,
        )
        report["trials"][name] = {
            "run_id": run_id,
            "override": patch,
            "aggregate": _aggregate(paths),
        }

    if not args.skip_oracle:
        run_id = f"paper_oracle_{args.tag}_hloc_top100"
        cfg = load_yaml(args.hloc_top100_config)
        cfg["_config_path"] = str(args.hloc_top100_config)
        paths = _run_matrix(
            dataset_root=args.dataset_root,
            outputs_root=args.outputs_root,
            run_id=run_id,
            method="hloc-megaloc",
            retrieval_config=cfg,
            eval_config=eval_top100_cfg,
            threshold_m=args.threshold_m,
        )
        report["oracle"] = {
            "run_id": run_id,
            "aggregate": _aggregate(paths),
        }

    out_path = args.outputs_root / f"paper_ablation_{args.tag}_summary.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"summary: {out_path}")


if __name__ == "__main__":
    main()
