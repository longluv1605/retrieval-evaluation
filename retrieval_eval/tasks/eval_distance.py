from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from retrieval_eval.io.dataset import DatasetLayout, get_query_names
from retrieval_eval.io.outputs import OutputLayout, ensure_dirs, write_json, write_stage_status
from retrieval_eval.io.pairs import read_pairs
from retrieval_eval.io.poses import load_positions


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), p))


def _success_rate(values: list[float], threshold: float) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if v <= threshold) / len(values)


def run_eval_distance(
    dataset_root: Path,
    outputs_root: Path,
    dataset_name: str,
    scene: str,
    map_dir: str,
    query_set: str,
    retrieval_method: str,
    run_id: str,
    config: dict[str, Any],
    threshold_m: float,
) -> Path:
    k_values = list(config.get("k_values", [1, 5, 10, 20]))
    k_values = [int(k) for k in k_values]
    cdf_thresholds = [float(t) for t in config.get("cdf_thresholds_m", [1.0, 2.0, 3.0, 5.0])]

    data_layout = DatasetLayout(
        dataset_root=dataset_root,
        dataset=dataset_name,
        scene=scene,
        query_set=query_set,
        map_dir=map_dir,
    )
    out_layout = OutputLayout(
        outputs_root=outputs_root,
        dataset=dataset_name,
        scene=scene,
        map_dir=map_dir,
        query_set=query_set,
        retrieval_method=retrieval_method,
        run_id=run_id,
    )
    ensure_dirs(out_layout)

    write_stage_status(
        out_layout.meta_dir / "stage_status.json",
        stage="eval_distance",
        status="running",
        details={"threshold_m": threshold_m, "k_values": k_values},
    )

    pairs_path = out_layout.retrieval_dir / "pairs.txt"
    pairs = read_pairs(pairs_path)

    map_pos = load_positions(data_layout.map_root / "poses.txt")
    query_pos = load_positions(data_layout.query_root / "poses.txt")

    queries_total = get_query_names(data_layout)
    query_set_names = set(queries_total)

    hits = {k: 0 for k in k_values}
    evaluated = 0
    missing_pose = 0
    top1_dist = []

    per_query_path = out_layout.eval_dir / "per_query.jsonl"
    pairs_scored_path = out_layout.eval_dir / "pairs_scored.csv"

    with per_query_path.open("w", encoding="utf-8") as perq, pairs_scored_path.open(
        "w", encoding="utf-8", newline=""
    ) as scored_file:
        writer = csv.writer(scored_file)
        writer.writerow([
            "query_name",
            "map_name",
            "score",
            "rank",
            "distance_m",
            "is_positive",
            "within_topk",
        ])

        for qname in queries_total:
            if qname not in query_set_names:
                continue
            if qname not in query_pos:
                missing_pose += 1
                perq.write(
                    json.dumps(
                        {"query_name": qname, "evaluated": False, "reason": "missing_query_pose"},
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                continue

            qpairs = pairs.get(qname, [])
            if not qpairs:
                evaluated += 1
                out = {
                    "query_name": qname,
                    "evaluated": True,
                    "best_rank_positive": None,
                    "hit_at": {str(k): False for k in k_values},
                    "top1_distance_m": None,
                }
                perq.write(json.dumps(out, ensure_ascii=True) + "\n")
                continue

            qpos = query_pos[qname]
            pos_ranks: list[int] = []
            top1 = None
            for mname, score, rank in qpairs:
                if mname not in map_pos:
                    continue
                dist = float(np.linalg.norm(qpos - map_pos[mname]))
                if top1 is None and rank == 1:
                    top1 = dist
                is_pos = int(dist <= threshold_m)
                if is_pos:
                    pos_ranks.append(rank)
                writer.writerow([qname, mname, f"{score:.8f}", rank, f"{dist:.6f}", is_pos, 1])

            evaluated += 1
            if top1 is not None:
                top1_dist.append(top1)

            best_rank = min(pos_ranks) if pos_ranks else None
            hit_map = {}
            for k in k_values:
                hit = best_rank is not None and best_rank <= k
                hit_map[str(k)] = bool(hit)
                if hit:
                    hits[k] += 1

            out = {
                "query_name": qname,
                "evaluated": True,
                "best_rank_positive": best_rank,
                "hit_at": hit_map,
                "top1_distance_m": top1,
            }
            perq.write(json.dumps(out, ensure_ascii=True) + "\n")

    recall = {f"R@{k}": (hits[k] / max(1, evaluated)) for k in k_values}

    # Top-1 CDF: success rate at each distance threshold
    top1_cdf = {}
    for tau in cdf_thresholds:
        top1_cdf[f"success_leq_{tau:.0f}m"] = _success_rate(top1_dist, tau)

    retrieval_cfg_path = out_layout.meta_dir / "retrieval_config.json"
    retrieval_cfg = {}
    if retrieval_cfg_path.exists():
        retrieval_cfg = json.loads(retrieval_cfg_path.read_text(encoding="utf-8"))

    evaluation = {
        "dataset": dataset_name,
        "scene": scene,
        "map_dir": map_dir,
        "query_set": query_set,
        "retrieval_method": retrieval_method,
        "run_id": run_id,
        "retrieval_config": retrieval_cfg,
        "distance_threshold_m": float(threshold_m),
        "num_queries_total": len(queries_total),
        "num_queries_evaluated": evaluated,
        "num_queries_missing_pose": missing_pose,
        "k_values": k_values,
        "recall": recall,
        "distance_stats_top1": {
            "mean": float(mean(top1_dist)) if top1_dist else 0.0,
            "median": float(median(top1_dist)) if top1_dist else 0.0,
            "p50": _percentile(top1_dist, 50.0),
            "p75": _percentile(top1_dist, 75.0),
            "p90": _percentile(top1_dist, 90.0),
            "p95": _percentile(top1_dist, 95.0),
            "p99": _percentile(top1_dist, 99.0),
        },
        "top1_cdf": top1_cdf,
        "cdf_thresholds_m": cdf_thresholds,
        "num_top1_samples": len(top1_dist),
        "files": {
            "pairs": "retrieval/pairs.txt",
            "per_query": "eval/per_query.jsonl",
            "pairs_scored": "eval/pairs_scored.csv",
        },
    }

    out_json = out_layout.eval_dir / "evaluation.json"
    write_json(out_json, evaluation)

    write_stage_status(
        out_layout.meta_dir / "stage_status.json",
        stage="eval_distance",
        status="done",
        details={"evaluation_json": str(out_json), "num_queries_evaluated": evaluated},
    )

    return out_json
