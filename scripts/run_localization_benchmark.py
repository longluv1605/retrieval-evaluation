#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
LAMAR_REPO = ROOT / "code/lamar-benchmark"
CROCODL_REPO = ROOT / "code/crocodl-benchmark"
DEFAULT_LAMAR_CAPTURES = "/home/robotics/VisualLocalization/capture_croco/lamar"
DEFAULT_CROCODL_CAPTURES = "/home/robotics/VisualLocalization/capture_croco"


@dataclass(frozen=True)
class Job:
    benchmark: str
    scene: str
    ref_id: str
    query_id: str
    query_filename: str
    is_rig: bool = False


DEFAULT_JOBS = [
    # CroCoDL challenge submission: ios->ios, 2 scenes (HYDRO + SUCCULENT).
    # is_rig=False because the iOS query is a single-camera device.
    Job("crocodl", "HYDRO", "ios_map", "ios_query", "keyframes_pruned_subsampled.txt", False),
    Job("crocodl", "SUCCULENT", "ios_map", "ios_query", "keyframes_pruned_subsampled.txt", False),
    # --- other scenes/devices (kept for reference) ---
    # Job("lamar", "HGE", "map", "query_val_phone", "queries.txt"),
    # Job("crocodl", "ARCHE_D2", "hl_map", "spot_query", "keyframes_original.txt", True),
    # Job("crocodl", "ARCHE_D2", "ios_map", "spot_query", "keyframes_original.txt", True),
    # Job("crocodl", "ARCHE_D2", "spot_map", "spot_query", "keyframes_original.txt", True),
]


def _repo_for(job: Job) -> Path:
    return LAMAR_REPO if job.benchmark == "lamar" else CROCODL_REPO


def _captures_for(job: Job, args: argparse.Namespace) -> Path:
    return args.lamar_captures if job.benchmark == "lamar" else args.crocodl_captures


def _config_for(job: Job, args: argparse.Namespace) -> Path:
    return args.lamar_seqfg_config if job.benchmark == "lamar" else args.crocodl_seqfg_config


def _results_path(out_root: Path, job: Job, method: str) -> Path:
    name = f"{job.benchmark}_{job.scene}_{job.ref_id}_{job.query_id}_{method}.json"
    return out_root / "raw" / name


def _run_command(cmd: list[str], cwd: Path, log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text("DRY RUN:\n" + " ".join(cmd) + "\n", encoding="utf-8")
        return 0

    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(proc.returncode)


def _baseline_command(job: Job, captures: Path, outputs: Path, results_json: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "lamar.run",
        "--scene",
        job.scene,
        "--ref_id",
        job.ref_id,
        "--query_id",
        job.query_id,
        "--captures",
        str(captures),
        "--outputs",
        str(outputs),
        "--retrieval",
        "megaloc",
        "--feature",
        "superpoint",
        "--matcher",
        "lightglue",
        # No --sequence_length_seconds: challenge queries have no odometry, so the
        # baseline runs single-image (per-frame PnP), matching the seqfg backend.
        "--query_filename",
        job.query_filename,
        "--results_json",
        str(results_json),
    ]
    if job.is_rig:
        cmd.append("--is_rig")
    return cmd


def _seqfg_command(
    job: Job,
    captures: Path,
    outputs: Path,
    config: Path,
    results_json: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "lamar.run_seqfg",
        "--scene",
        job.scene,
        "--ref_id",
        job.ref_id,
        "--query_id",
        job.query_id,
        "--captures",
        str(captures),
        "--outputs",
        str(outputs),
        "--config",
        str(config),
        "--query_filename",
        job.query_filename,
        "--results_json",
        str(results_json),
    ]
    if job.is_rig:
        cmd.append("--is_rig")
    return cmd


def _select_recall(payload: dict[str, Any]) -> tuple[float | None, float | None, str]:
    recall = payload.get("recall")
    if isinstance(recall, dict):
        values = recall.get("final") or recall.get("reloc") or recall.get("loc")
        stage = "final" if recall.get("final") is not None else "reloc" if recall.get("reloc") is not None else "loc"
    else:
        values = recall
        stage = "single_image"

    if not isinstance(values, list) or len(values) < 2:
        return None, None, stage
    return float(values[0]), float(values[1]), stage


def _select_count(payload: dict[str, Any], key: str, stage: str) -> int | None:
    value = payload.get(key)
    if isinstance(value, dict):
        value = value.get(stage)
    if value is None:
        return None
    return int(value)


def _row_from_result(job: Job, method: str, path: Path, returncode: int, skipped: bool) -> dict[str, Any]:
    base = {
        "benchmark": job.benchmark,
        "scene": job.scene,
        "ref_id": job.ref_id,
        "query_id": job.query_id,
        "method": method,
        "status": "skipped" if skipped else "failed" if returncode != 0 else "ok",
        "returncode": returncode,
        "stage": "",
        "pose_recall_10cm_1deg": None,
        "pose_recall_1m_5deg": None,
        "num_queries": None,
        "num_localized": None,
        "num_failed": None,
        "retrieval_R@10": None,
        "runtime_s_per_query": None,
        "results_json": str(path),
    }
    if returncode != 0 or skipped or not path.exists():
        return base

    payload = json.loads(path.read_text(encoding="utf-8"))
    r_strict, r_loose, stage = _select_recall(payload)
    base.update(
        {
            "stage": stage,
            "pose_recall_10cm_1deg": r_strict,
            "pose_recall_1m_5deg": r_loose,
            "num_queries": _select_count(payload, "num_queries", stage),
            "num_localized": _select_count(payload, "num_localized", stage),
            "num_failed": _select_count(payload, "num_failed", stage),
        }
    )
    return base


def _write_outputs(out_root: Path, rows: list[dict[str, Any]]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
    }
    (out_root / "localization_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    fields = list(rows[0].keys()) if rows else []
    with (out_root / "localization_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline and SeqFG localization benchmarks.")
    parser.add_argument("--outputs-root", type=Path, default=ROOT / "outputs_localization")
    parser.add_argument("--lamar-captures", type=Path, default=DEFAULT_LAMAR_CAPTURES)
    parser.add_argument("--crocodl-captures", type=Path, default=DEFAULT_CROCODL_CAPTURES)
    parser.add_argument(
        "--lamar-seqfg-config",
        type=Path,
        default=LAMAR_REPO / "lamar/configs/seqfg_hge_default.yaml",
    )
    parser.add_argument(
        "--crocodl-seqfg-config",
        type=Path,
        default=CROCODL_REPO / "lamar/configs/seqfg_default.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for job in DEFAULT_JOBS:
        repo = _repo_for(job)
        captures = _captures_for(job, args)
        outputs = args.outputs_root / job.benchmark
        scene_path = captures / job.scene
        skipped = not scene_path.exists()
        for method in ("independent", "seqfg_time"):
            result_path = _results_path(args.outputs_root, job, method)
            log_path = args.outputs_root / "logs" / f"{result_path.stem}.log"
            if skipped:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(f"Skipped: missing capture scene {scene_path}\n", encoding="utf-8")
                returncode = 0
            elif method == "independent":
                cmd = _baseline_command(job, captures, outputs, result_path)
                returncode = _run_command(cmd, repo, log_path, args.dry_run)
            else:
                cmd = _seqfg_command(job, captures, outputs, _config_for(job, args), result_path)
                returncode = _run_command(cmd, repo, log_path, args.dry_run)
            rows.append(_row_from_result(job, method, result_path, returncode, skipped))

    _write_outputs(args.outputs_root, rows)
    print(f"summary: {args.outputs_root / 'localization_summary.json'}")
    print(f"table: {args.outputs_root / 'localization_summary.csv'}")


if __name__ == "__main__":
    main()
