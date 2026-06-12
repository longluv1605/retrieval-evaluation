from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunKey:
    dataset: str
    scene: str
    map_dir: str
    query_set: str


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_per_query(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rows[str(rec["query_name"])] = rec
    return rows


def _read_pairs_scored(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f):
            rec["rank"] = int(rec["rank"])
            rec["score"] = float(rec["score"])
            rec["distance_m"] = float(rec["distance_m"])
            rec["is_positive"] = int(rec["is_positive"])
            rows[str(rec["query_name"])].append(rec)
    for qrows in rows.values():
        qrows.sort(key=lambda r: r["rank"])
    return dict(rows)


def _find_evaluations(outputs_root: Path, run_id: str, method: str) -> dict[RunKey, Path]:
    root = outputs_root / run_id
    found: dict[RunKey, Path] = {}
    for path in root.glob("**/eval/evaluation.json"):
        try:
            data = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if str(data.get("retrieval_method", "")) != method:
            continue
        key = RunKey(
            dataset=str(data.get("dataset", "")),
            scene=str(data.get("scene", "")),
            map_dir=str(data.get("map_dir", "")),
            query_set=str(data.get("query_set", "")),
        )
        found[key] = path
    return found


def _best_positive_rank(qrows: list[dict[str, Any]], max_rank: int | None = None) -> int | None:
    ranks = [
        int(r["rank"])
        for r in qrows
        if int(r["is_positive"]) == 1 and (max_rank is None or int(r["rank"]) <= max_rank)
    ]
    return min(ranks) if ranks else None


def _hit_at(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def analyze_pair(
    base_eval_path: Path,
    seq_eval_path: Path,
    oracle_k: int,
) -> dict[str, Any]:
    base_eval = _read_json(base_eval_path)
    seq_eval = _read_json(seq_eval_path)

    base_perq = _read_per_query(base_eval_path.parent / "per_query.jsonl")
    seq_perq = _read_per_query(seq_eval_path.parent / "per_query.jsonl")
    base_scored = _read_pairs_scored(base_eval_path.parent / "pairs_scored.csv")
    seq_scored = _read_pairs_scored(seq_eval_path.parent / "pairs_scored.csv")

    common_queries = sorted(set(base_perq) & set(seq_perq))
    evaluated = [
        q
        for q in common_queries
        if bool(base_perq[q].get("evaluated")) and bool(seq_perq[q].get("evaluated"))
    ]

    counts = {
        "evaluated_common": len(evaluated),
        "baseline_top1_hit": 0,
        "seq_top1_hit": 0,
        "top1_promoted": 0,
        "top1_lost": 0,
        "top10_promoted": 0,
        "top10_lost": 0,
        "oracle_hit": 0,
        "seq_failure_oracle_miss": 0,
        "seq_failure_oracle_hit": 0,
    }

    promotion_examples: list[dict[str, Any]] = []
    failure_examples: list[dict[str, Any]] = []

    for q in evaluated:
        base_rank = base_perq[q].get("best_rank_positive")
        seq_rank = seq_perq[q].get("best_rank_positive")
        base_rank = int(base_rank) if base_rank is not None else None
        seq_rank = int(seq_rank) if seq_rank is not None else None

        base_top1 = _hit_at(base_rank, 1)
        seq_top1 = _hit_at(seq_rank, 1)
        base_top10 = _hit_at(base_rank, 10)
        seq_top10 = _hit_at(seq_rank, 10)

        counts["baseline_top1_hit"] += int(base_top1)
        counts["seq_top1_hit"] += int(seq_top1)
        counts["top1_promoted"] += int((not base_top1) and seq_top1)
        counts["top1_lost"] += int(base_top1 and (not seq_top1))
        counts["top10_promoted"] += int((not base_top10) and seq_top10)
        counts["top10_lost"] += int(base_top10 and (not seq_top10))

        oracle_rank = _best_positive_rank(base_scored.get(q, []), max_rank=oracle_k)
        oracle_hit = _hit_at(oracle_rank, oracle_k)
        counts["oracle_hit"] += int(oracle_hit)
        if not seq_top1:
            counts["seq_failure_oracle_miss"] += int(not oracle_hit)
            counts["seq_failure_oracle_hit"] += int(oracle_hit)

        if (not base_top1) and seq_top1 and len(promotion_examples) < 20:
            promotion_examples.append(
                {
                    "query_name": q,
                    "baseline_best_rank_positive": base_rank,
                    "seq_best_rank_positive": seq_rank,
                    "baseline_top1_distance_m": base_perq[q].get("top1_distance_m"),
                    "seq_top1_distance_m": seq_perq[q].get("top1_distance_m"),
                    "baseline_oracle_rank": oracle_rank,
                }
            )
        if base_top1 and (not seq_top1) and len(failure_examples) < 20:
            failure_examples.append(
                {
                    "query_name": q,
                    "baseline_best_rank_positive": base_rank,
                    "seq_best_rank_positive": seq_rank,
                    "baseline_top1_distance_m": base_perq[q].get("top1_distance_m"),
                    "seq_top1_distance_m": seq_perq[q].get("top1_distance_m"),
                    "baseline_oracle_rank": oracle_rank,
                }
            )

    denom = max(1, counts["evaluated_common"])
    rates = {
        "baseline_R@1": counts["baseline_top1_hit"] / denom,
        "seq_R@1": counts["seq_top1_hit"] / denom,
        "delta_R@1": (counts["seq_top1_hit"] - counts["baseline_top1_hit"]) / denom,
        f"baseline_oracle_R@{oracle_k}": counts["oracle_hit"] / denom,
        "top1_promotion_rate": counts["top1_promoted"] / denom,
        "top1_loss_rate": counts["top1_lost"] / denom,
    }

    return {
        "dataset": base_eval["dataset"],
        "scene": base_eval["scene"],
        "map_dir": base_eval["map_dir"],
        "query_set": base_eval["query_set"],
        "baseline_method": base_eval["retrieval_method"],
        "sequence_method": seq_eval["retrieval_method"],
        "oracle_k": oracle_k,
        "counts": counts,
        "rates": rates,
        "promotion_examples": promotion_examples,
        "failure_examples": failure_examples,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], oracle_k: int) -> None:
    fields = [
        "dataset",
        "scene",
        "map_dir",
        "query_set",
        "baseline_method",
        "sequence_method",
        "evaluated_common",
        "baseline_R@1",
        "seq_R@1",
        "delta_R@1",
        f"baseline_oracle_R@{oracle_k}",
        "top1_promoted",
        "top1_lost",
        "seq_failure_oracle_miss",
        "seq_failure_oracle_hit",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {
                "dataset": row["dataset"],
                "scene": row["scene"],
                "map_dir": row["map_dir"],
                "query_set": row["query_set"],
                "baseline_method": row["baseline_method"],
                "sequence_method": row["sequence_method"],
                "evaluated_common": row["counts"]["evaluated_common"],
                "baseline_R@1": row["rates"]["baseline_R@1"],
                "seq_R@1": row["rates"]["seq_R@1"],
                "delta_R@1": row["rates"]["delta_R@1"],
                f"baseline_oracle_R@{oracle_k}": row["rates"][f"baseline_oracle_R@{oracle_k}"],
                "top1_promoted": row["counts"]["top1_promoted"],
                "top1_lost": row["counts"]["top1_lost"],
                "seq_failure_oracle_miss": row["counts"]["seq_failure_oracle_miss"],
                "seq_failure_oracle_hit": row["counts"]["seq_failure_oracle_hit"],
            }
            writer.writerow(flat)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze reranking promotions, losses, and candidate-pool oracle recall."
    )
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--baseline-method", type=str, default="hloc-megaloc")
    parser.add_argument("--sequence-method", type=str, default="seqfg")
    parser.add_argument("--oracle-k", type=int, default=100)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    baseline = _find_evaluations(args.outputs_root, args.run_id, args.baseline_method)
    sequence = _find_evaluations(args.outputs_root, args.run_id, args.sequence_method)
    keys = sorted(set(baseline) & set(sequence), key=lambda k: (k.dataset, k.scene, k.map_dir, k.query_set))
    if not keys:
        raise SystemExit(
            "[ERROR] no matching evaluation pairs found for "
            f"{args.baseline_method!r} and {args.sequence_method!r}"
        )

    rows = [analyze_pair(baseline[key], sequence[key], oracle_k=args.oracle_k) for key in keys]
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=True), encoding="utf-8")
    write_csv(args.out_csv, rows, oracle_k=args.oracle_k)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
