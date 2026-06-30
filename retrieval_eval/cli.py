from __future__ import annotations

import argparse
import json
from pathlib import Path

from retrieval_eval.config import load_yaml
from retrieval_eval.run import run
from retrieval_eval.tasks.pair_selection import PairSelection
from retrieval_eval.tasks.eval_distance import run_eval_distance
from retrieval_eval.tasks.retrieve import run_retrieve


def _common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--map-dir", type=str, default="map")
    parser.add_argument("--query-set", type=str, required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--retrieval-method",
        type=str,
        required=True,
        choices=list(PairSelection.methods),
    )
    parser.add_argument("--run-id", type=str, required=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="retrieval-eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ret = sub.add_parser("retrieve", help="Run retrieval and produce pairs.txt")
    _common_args(p_ret)
    p_ret.add_argument("--config", type=Path, required=True)

    p_eval = sub.add_parser("eval-distance", help="Evaluate retrieval using distance threshold")
    _common_args(p_eval)
    p_eval.add_argument("--config", type=Path, required=True)
    p_eval.add_argument("--threshold-m", type=float, required=True)

    p_run = sub.add_parser("run", help="Run retrieval then eval-distance")
    _common_args(p_run)
    p_run.add_argument("--retrieval-config", type=Path, required=True)
    p_run.add_argument("--eval-config", type=Path, required=True)
    p_run.add_argument("--threshold-m", type=float, required=True)

    args = parser.parse_args()

    if args.command == "retrieve":
        cfg = load_yaml(args.config)
        cfg["_config_path"] = str(args.config)
        pairs = run_retrieve(
            dataset_root=args.dataset_root,
            outputs_root=args.outputs_root,
            dataset_name=args.dataset,
            scene=args.scene,
            map_dir=args.map_dir,
            query_set=args.query_set,
            retrieval_method=args.retrieval_method,
            run_id=args.run_id,
            config=cfg,
        )
        print(f"pairs saved: {pairs}")
        return

    if args.command == "eval-distance":
        cfg = load_yaml(args.config)
        cfg["_config_path"] = str(args.config)
        out_json = run_eval_distance(
            dataset_root=args.dataset_root,
            outputs_root=args.outputs_root,
            dataset_name=args.dataset,
            scene=args.scene,
            map_dir=args.map_dir,
            query_set=args.query_set,
            retrieval_method=args.retrieval_method,
            run_id=args.run_id,
            config=cfg,
            threshold_m=float(args.threshold_m),
        )
        print(f"evaluation saved: {out_json}")
        return

    if args.command == "run":
        retrieval_cfg = load_yaml(args.retrieval_config)
        retrieval_cfg["_config_path"] = str(args.retrieval_config)
        eval_cfg = load_yaml(args.eval_config)
        eval_cfg["_config_path"] = str(args.eval_config)
        pairs_path, eval_path = run(
            dataset_root=args.dataset_root,
            outputs_root=args.outputs_root,
            dataset_name=args.dataset,
            scene=args.scene,
            map_dir=args.map_dir,
            query_set=args.query_set,
            retrieval_method=args.retrieval_method,
            run_id=args.run_id,
            retrieval_config=retrieval_cfg,
            eval_config=eval_cfg,
            threshold_m=float(args.threshold_m),
        )
        print(f"pairs saved: {pairs_path}")
        print(f"evaluation saved: {eval_path}")
        return


if __name__ == "__main__":
    main()
