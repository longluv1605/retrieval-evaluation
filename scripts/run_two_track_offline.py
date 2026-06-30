#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval_eval.config import load_yaml
from retrieval_eval.run import run


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _method_alias(method_name: str) -> str:
    method = method_name.strip().lower()
    if method == "segfg":
        return "seqfg"
    return method


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _apply_trial_override(base_cfg: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    override = {k: v for k, v in override.items() if k != "name"}
    if "retrieval" in override:
        return _deep_merge(base_cfg, override)

    cfg = copy.deepcopy(base_cfg)
    retrieval = dict(cfg.get("retrieval", {}))
    seq = dict(retrieval.get("sequence", {}))

    retrieval_keys = {"top_k", "global_model", "overwrite_features", "sequence"}
    seq_patch: dict[str, Any] = {}
    for key, value in override.items():
        if key in retrieval_keys:
            retrieval[key] = value
        else:
            seq_patch[key] = value

    seq.update(seq_patch)
    retrieval["sequence"] = seq
    cfg["retrieval"] = retrieval
    return cfg


def _load_trial_overrides(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return [{}]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Trial overrides JSON must be a non-empty list of objects.")
    parsed: list[dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Trial override at index {idx} must be an object.")
        parsed.append(item)
    return parsed


def _recall_triplet(eval_payload: dict[str, Any]) -> tuple[float, float, float]:
    recall = dict(eval_payload.get("recall", {}))
    return (
        float(recall.get("R@1", 0.0)),
        float(recall.get("R@5", 0.0)),
        float(recall.get("R@10", 0.0)),
    )


def _make_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class SceneSpec:
    dataset: str
    scene: str
    map_dir: str
    query_set: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two-track offline protocol: baseline-fresh first, then per-scene segfg/seqfg tuning."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--map-dir", type=str, default="map")
    parser.add_argument("--query-set", type=str, required=True)

    parser.add_argument("--baseline-method", type=str, default="hloc-megaloc")
    parser.add_argument("--segfg-method", type=str, default="segfg")

    parser.add_argument("--baseline-retrieval-config", type=Path, default=Path("configs/retrieval.hloc.yaml"))
    parser.add_argument("--segfg-retrieval-config", type=Path, default=Path("configs/retrieval.sequence.yaml"))
    parser.add_argument("--eval-config", type=Path, default=Path("configs/eval.yaml"))

    parser.add_argument("--threshold-m", type=float, default=1.0)
    parser.add_argument("--trial-overrides-json", type=Path, default=None)
    parser.add_argument("--tag", type=str, default="")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    tag = args.tag.strip() or _now_tag()
    baseline_method = _method_alias(args.baseline_method)
    segfg_method = _method_alias(args.segfg_method)
    scene = SceneSpec(
        dataset=args.dataset,
        scene=args.scene,
        map_dir=args.map_dir,
        query_set=args.query_set,
    )

    baseline_cfg = load_yaml(args.baseline_retrieval_config)
    baseline_cfg["_config_path"] = str(args.baseline_retrieval_config)

    segfg_base_cfg = load_yaml(args.segfg_retrieval_config)
    segfg_base_cfg["_config_path"] = str(args.segfg_retrieval_config)

    eval_cfg = load_yaml(args.eval_config)
    eval_cfg["_config_path"] = str(args.eval_config)

    trial_overrides = _load_trial_overrides(args.trial_overrides_json)

    tuned_root = (
        args.outputs_root
        / "tuned_outputs"
        / scene.dataset
        / scene.scene
        / scene.map_dir
        / scene.query_set
    )
    compare_dir = tuned_root / "compare"
    selection_dir = tuned_root / "selection"
    compare_dir.mkdir(parents=True, exist_ok=True)
    selection_dir.mkdir(parents=True, exist_ok=True)

    baseline_run_id = f"baseline_fresh_{tag}"
    print(f"[1/3] Running baseline fresh: {baseline_method} ({baseline_run_id})")
    _, baseline_eval_path = run(
        dataset_root=args.dataset_root,
        outputs_root=args.outputs_root,
        dataset_name=scene.dataset,
        scene=scene.scene,
        map_dir=scene.map_dir,
        query_set=scene.query_set,
        retrieval_method=baseline_method,
        run_id=baseline_run_id,
        retrieval_config=baseline_cfg,
        eval_config=eval_cfg,
        threshold_m=float(args.threshold_m),
    )
    baseline_eval = _read_json(baseline_eval_path)
    base_r1, base_r5, base_r10 = _recall_triplet(baseline_eval)

    print(f"[2/3] Running segfg/seqfg tuning trials: {len(trial_overrides)}")
    trials_report: list[dict[str, Any]] = []
    for idx, override in enumerate(trial_overrides, start=1):
        trial_name = str(override.get("name", f"trial_{idx:03d}"))
        trial_cfg = _apply_trial_override(segfg_base_cfg, override)
        trial_run_id = f"segfg_tuned_{tag}_{idx:03d}"
        _, trial_eval_path = run(
            dataset_root=args.dataset_root,
            outputs_root=args.outputs_root,
            dataset_name=scene.dataset,
            scene=scene.scene,
            map_dir=scene.map_dir,
            query_set=scene.query_set,
            retrieval_method=segfg_method,
            run_id=trial_run_id,
            retrieval_config=trial_cfg,
            eval_config=eval_cfg,
            threshold_m=float(args.threshold_m),
        )
        trial_eval = _read_json(trial_eval_path)
        r1, r5, r10 = _recall_triplet(trial_eval)

        d1 = r1 - base_r1
        d5 = r5 - base_r5
        d10 = r10 - base_r10
        pass_all = d1 >= 0.0 and d5 >= 0.0 and d10 >= 0.0
        score = 0.60 * d1 + 0.25 * d5 + 0.15 * d10

        trials_report.append(
            {
                "trial_index": idx,
                "trial_name": trial_name,
                "run_id": trial_run_id,
                "override": override,
                "metrics": {"R@1": r1, "R@5": r5, "R@10": r10},
                "delta_vs_baseline": {"R@1": d1, "R@5": d5, "R@10": d10},
                "pass_all_metrics": pass_all,
                "score": score,
                "evaluation_json": _make_relative(trial_eval_path, args.outputs_root),
            }
        )

    pass_trials = [t for t in trials_report if t["pass_all_metrics"]]
    if pass_trials:
        selected = max(pass_trials, key=lambda x: (x["score"], x["delta_vs_baseline"]["R@1"]))
        verdict = "pass"
        reason = "Selected best scoring trial among runs that beat baseline on R@1/R@5/R@10."
    else:
        selected = max(trials_report, key=lambda x: (x["score"], x["delta_vs_baseline"]["R@1"]))
        verdict = "review"
        reason = "No trial beat baseline on all R@1/R@5/R@10; selected highest score for analysis track only."

    manifest = {
        "protocol": "two_track_offline_baseline_first",
        "tag": tag,
        "scene": {
            "dataset": scene.dataset,
            "scene": scene.scene,
            "map_dir": scene.map_dir,
            "query_set": scene.query_set,
        },
        "inputs": {
            "dataset_root": str(args.dataset_root),
            "outputs_root": str(args.outputs_root),
            "baseline_retrieval_config": str(args.baseline_retrieval_config),
            "segfg_retrieval_config": str(args.segfg_retrieval_config),
            "eval_config": str(args.eval_config),
            "threshold_m": float(args.threshold_m),
            "trial_overrides_json": str(args.trial_overrides_json) if args.trial_overrides_json else None,
        },
        "baseline_fresh": {
            "method": baseline_method,
            "run_id": baseline_run_id,
            "evaluation_json": _make_relative(baseline_eval_path, args.outputs_root),
            "metrics": {"R@1": base_r1, "R@5": base_r5, "R@10": base_r10},
        },
        "analysis_track": {
            "method": segfg_method,
            "num_trials": len(trials_report),
        },
    }
    _write_json(tuned_root / "study_manifest.json", manifest)

    compare_report = {
        "baseline_run_id": baseline_run_id,
        "selection_verdict": verdict,
        "selection_reason": reason,
        "trials": trials_report,
    }
    _write_json(compare_dir / f"compare_{tag}.json", compare_report)

    selected_report = {
        "verdict": verdict,
        "reason": reason,
        "selected": selected,
        "baseline": {
            "run_id": baseline_run_id,
            "metrics": {"R@1": base_r1, "R@5": base_r5, "R@10": base_r10},
            "evaluation_json": _make_relative(baseline_eval_path, args.outputs_root),
        },
    }
    _write_json(selection_dir / f"selected_{tag}.json", selected_report)

    md_lines = [
        "# Two-Track Offline Summary",
        "",
        f"- Scene: {scene.dataset}/{scene.scene}/{scene.map_dir}/{scene.query_set}",
        f"- Tag: {tag}",
        f"- Baseline run: {baseline_run_id}",
        f"- Verdict: {verdict}",
        f"- Reason: {reason}",
        "",
        "## Baseline",
        f"- R@1={base_r1:.6f}, R@5={base_r5:.6f}, R@10={base_r10:.6f}",
        "",
        "## Selected Trial",
        f"- Run ID: {selected['run_id']}",
        (
            "- Delta vs baseline: "
            f"R@1={selected['delta_vs_baseline']['R@1']:+.6f}, "
            f"R@5={selected['delta_vs_baseline']['R@5']:+.6f}, "
            f"R@10={selected['delta_vs_baseline']['R@10']:+.6f}"
        ),
    ]
    (selection_dir / f"summary_{tag}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("[3/3] Wrote tuned outputs artifacts")
    print(f"manifest: {tuned_root / 'study_manifest.json'}")
    print(f"compare : {compare_dir / f'compare_{tag}.json'}")
    print(f"select  : {selection_dir / f'selected_{tag}.json'}")


if __name__ == "__main__":
    main()
