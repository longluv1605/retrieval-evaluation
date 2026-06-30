#!/usr/bin/env python3
"""Generate Chapter 4 thesis artifacts from a benchmark run.

Inputs:
- summary_recall.csv (pairwise recall summary)
- run_root containing evaluation.json / per_query.jsonl

Outputs:
- Final-Thesis/chapters/ch4_tables.tex
- Final-Thesis/chapters/appendix_case_trace.tex
- Final-Thesis/images/fig_ch4_*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class PairRow:
    dataset: str
    scene: str
    map_dir: str
    query_set: str
    hloc_r1: float
    seqfg_r1: float
    delta_r1: float
    hloc_r5: float
    seqfg_r5: float
    delta_r5: float
    hloc_r10: float
    seqfg_r10: float
    delta_r10: float
    winner: str


@dataclass(frozen=True)
class EvalStats:
    mean: float
    median: float
    p90: float
    p95: float
    r1: float
    r5: float
    r10: float
    num_queries: int
    cdf_1m: float
    cdf_2m: float
    cdf_3m: float
    cdf_5m: float


@dataclass(frozen=True)
class CaseTrace:
    case_id: str
    query_name: str
    baseline_top1: float
    seqfg_top1: float
    baseline_hit: bool
    seqfg_hit: bool
    note: str


def fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}"


def fmt_pct_signed(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.2f}"


def fmt_float(v: float) -> str:
    return f"{v:.3f}"


def latex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = text
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def parse_summary(summary_csv: Path) -> List[PairRow]:
    rows: List[PairRow] = []
    with summary_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                PairRow(
                    dataset=row["dataset"],
                    scene=row["scene"],
                    map_dir=row["map_dir"],
                    query_set=row["query_set"],
                    hloc_r1=float(row["hloc_R@1"]),
                    seqfg_r1=float(row["seqfg_R@1"]),
                    delta_r1=float(row["delta_R@1"]),
                    hloc_r5=float(row["hloc_R@5"]),
                    seqfg_r5=float(row["seqfg_R@5"]),
                    delta_r5=float(row["delta_R@5"]),
                    hloc_r10=float(row["hloc_R@10"]),
                    seqfg_r10=float(row["seqfg_R@10"]),
                    delta_r10=float(row["delta_R@10"]),
                    winner=row["winner"],
                )
            )
    return rows


def parse_eval_files(run_root: Path) -> dict[tuple[str, str, str, str, str], dict[str, object]]:
    evals: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    for path in run_root.glob("**/eval/evaluation.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        key = (
            str(data["dataset"]),
            str(data["scene"]),
            str(data["map_dir"]),
            str(data["query_set"]),
            str(data["retrieval_method"]),
        )
        evals[key] = data
    return evals


def parse_per_query(run_root: Path) -> dict[tuple[str, str, str, str], dict[str, dict[str, dict[str, object]]]]:
    out: dict[tuple[str, str, str, str], dict[str, dict[str, dict[str, object]]]] = defaultdict(dict)
    for path in run_root.glob("**/eval/per_query.jsonl"):
        parts = path.relative_to(run_root).parts
        method = parts[-3]
        query_set = parts[-4]
        map_dir = parts[-5]
        scene = parts[-6]
        dataset = "/".join(parts[:-6])
        pair_key = (dataset, scene, map_dir, query_set)
        records: dict[str, dict[str, object]] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                records[str(rec["query_name"])] = rec
        out[pair_key][method] = records
    return out


def compute_stats(evals: dict[tuple[str, str, str, str, str], dict[str, object]]) -> dict[str, EvalStats]:
    by_method: dict[str, dict[str, float]] = defaultdict(lambda: {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "r1": 0.0, "r5": 0.0, "r10": 0.0, "cdf_1m": 0.0, "cdf_2m": 0.0, "cdf_3m": 0.0, "cdf_5m": 0.0, "n": 0})
    for data in evals.values():
        method = str(data["retrieval_method"])
        n = int(data["num_queries_evaluated"])
        by_method[method]["n"] += n
        by_method[method]["mean"] += float(data["distance_stats_top1"]["mean"]) * n
        by_method[method]["median"] += float(data["distance_stats_top1"]["median"]) * n
        by_method[method]["p90"] += float(data["distance_stats_top1"]["p90"]) * n
        by_method[method]["p95"] += float(data["distance_stats_top1"].get("p95", 0.0)) * n
        by_method[method]["r1"] += float(data["recall"]["R@1"]) * n
        by_method[method]["r5"] += float(data["recall"]["R@5"]) * n
        by_method[method]["r10"] += float(data["recall"]["R@10"]) * n
        # CDF metrics (gracefully handle old evaluation.json without top1_cdf)
        top1_cdf = data.get("top1_cdf", {})
        for tau_key, acc_key in [("success_leq_1m", "cdf_1m"), ("success_leq_2m", "cdf_2m"), ("success_leq_3m", "cdf_3m"), ("success_leq_5m", "cdf_5m")]:
            by_method[method][acc_key] += float(top1_cdf.get(tau_key, 0.0)) * n

    out: dict[str, EvalStats] = {}
    for method, acc in by_method.items():
        n = int(acc["n"])
        out[method] = EvalStats(
            mean=acc["mean"] / n,
            median=acc["median"] / n,
            p90=acc["p90"] / n,
            p95=acc["p95"] / n,
            r1=acc["r1"] / n,
            r5=acc["r5"] / n,
            r10=acc["r10"] / n,
            num_queries=n,
            cdf_1m=acc["cdf_1m"] / n,
            cdf_2m=acc["cdf_2m"] / n,
            cdf_3m=acc["cdf_3m"] / n,
            cdf_5m=acc["cdf_5m"] / n,
        )
    return out


def pair_labels(rows: Sequence[PairRow]) -> list[str]:
    return [f"{r.dataset}/{r.scene}/{r.map_dir}/{r.query_set}" for r in rows]


def make_pairwise_recall_fig(rows: Sequence[PairRow], out_path: Path) -> None:
    labels = pair_labels(rows)
    y = np.arange(len(rows))
    h = 0.38
    fig, ax = plt.subplots(figsize=(14.5, max(7.5, len(rows) * 0.42)))
    ax.barh(y - h / 2, [r.hloc_r1 * 100 for r in rows], height=h, label="hloc-megaloc", color="#4C78A8")
    ax.barh(y + h / 2, [r.seqfg_r1 * 100 for r in rows], height=h, label="seqfg", color="#F58518")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Recall@1 (%)")
    ax.set_title("Recall@1 theo từng cặp dataset-scene-map-query")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_delta_fig(rows: Sequence[PairRow], out_path: Path) -> None:
    ordered = sorted(rows, key=lambda r: (r.delta_r1, r.dataset, r.scene, r.map_dir, r.query_set), reverse=True)
    labels = [f"{r.dataset}/{r.scene}/{r.map_dir}/{r.query_set}" for r in ordered]
    values = [r.delta_r1 * 100 for r in ordered]
    colors = ["#54A24B" if v >= 0 else "#E45756" for v in values]
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(14.5, max(7.5, len(rows) * 0.42)))
    ax.barh(y, values, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("ΔRecall@1 (seqfg - hloc-megaloc), điểm %")
    ax.set_title("Chênh lệch Recall@1 theo cặp dữ liệu")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_cdf_fig(run_root: Path, out_path: Path) -> None:
    dists = defaultdict(list)
    for path in run_root.glob("**/eval/per_query.jsonl"):
        method = path.relative_to(run_root).parts[-3]
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("evaluated") and rec.get("top1_distance_m") is not None:
                    dists[method].append(float(rec["top1_distance_m"]))
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for method, color in [("hloc-megaloc", "#4C78A8"), ("seqfg", "#F58518")]:
        vals = np.sort(np.asarray(dists[method], dtype=float))
        if len(vals) == 0:
            continue
        y = np.linspace(0, 1, len(vals), endpoint=True)
        ax.plot(vals, y, label=method, linewidth=2.4, color=color)
    # Add vertical threshold lines for CDF reference
    for tau in [1.0, 2.0, 3.0, 5.0]:
        ax.axvline(tau, color="grey", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.text(tau + 0.1, 0.02, f"{tau:.0f}m", fontsize=8, color="grey", alpha=0.7)
    ax.set_xlabel("Top-1 distance (m)")
    ax.set_ylabel("CDF")
    ax.set_xlim(0, 15)
    ax.set_title("CDF khoảng cách top-1 của hai phương pháp cốt lõi")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_recall_profile_fig(stats: dict[str, EvalStats], out_path: Path) -> None:
    metrics = ["R@1", "R@5", "R@10"]
    hloc_vals = [stats["hloc-megaloc"].r1 * 100, stats["hloc-megaloc"].r5 * 100, stats["hloc-megaloc"].r10 * 100]
    seq_vals = [stats["seqfg"].r1 * 100, stats["seqfg"].r5 * 100, stats["seqfg"].r10 * 100]
    x = np.arange(len(metrics))
    w = 0.32
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.bar(x - w / 2, hloc_vals, width=w, label="hloc-megaloc", color="#4C78A8")
    ax.bar(x + w / 2, seq_vals, width=w, label="seqfg", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Recall (%)")
    ax.set_title("Hồ sơ Recall@K của hai phương pháp cốt lõi")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_tables_tex(rows: Sequence[PairRow], stats: dict[str, EvalStats], out_path: Path) -> None:
    pair_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{So s\'anh theo c\$\?p \\texttt{dataset-scene-map-query} giữa \\texttt{hloc-megaloc} và \\texttt{seqfg}.}",
        r"\label{tab:ch4_core_pairwise_recall}",
        r"\small",
        r"\begingroup",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llllrrrrrrrrrr}",
        r"\hline",
        r"Dataset & Scene & Map & Query & HLOC R@1 & SeqFG R@1 & $\Delta$R@1 & HLOC R@5 & SeqFG R@5 & $\Delta$R@5 & HLOC R@10 & SeqFG R@10 & $\Delta$R@10 & Winner \\",
        r"\hline",
    ]
    for r in rows:
        pair_lines.append(
            f"{latex_escape(r.dataset)} & {latex_escape(r.scene)} & {latex_escape(r.map_dir)} & {latex_escape(r.query_set)} & "
            f"{fmt_pct(r.hloc_r1)} & {fmt_pct(r.seqfg_r1)} & {fmt_pct_signed(r.delta_r1)} & "
            f"{fmt_pct(r.hloc_r5)} & {fmt_pct(r.seqfg_r5)} & {fmt_pct_signed(r.delta_r5)} & "
            f"{fmt_pct(r.hloc_r10)} & {fmt_pct(r.seqfg_r10)} & {fmt_pct_signed(r.delta_r10)} & {latex_escape(r.winner)} \\")
    pair_lines += [
        r"\hline",
        r"\end{tabular}",
        r"}",
        r"\endgroup",
        r"\end{table*}",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Th\'ong k\^e t\'om h\?p c\^ua hai ph\u01b0\u01a1ng ph\'ap c\'ot l\u00f5i tr\^en t\`oan b\u1ed9 13{,}313 truy v\u1ea5n/\u0111\u01a1n v\u1ecb ph\u01b0\u01a1ng ph\'ap.}",
        r"\label{tab:ch4_core_distance_summary}",
        r"\small",
        r"\begin{tabular}{lrrrrrrr}",
        r"\hline",
        r"Method & Queries & Top1 mean (m) & Top1 median (m) & Top1 p90 (m) & R@1 & R@5 & R@10 \\",
        r"\hline",
        f"hloc-megaloc & {stats['hloc-megaloc'].num_queries:,} & {stats['hloc-megaloc'].mean:.3f} & {stats['hloc-megaloc'].median:.3f} & {stats['hloc-megaloc'].p90:.3f} & {stats['hloc-megaloc'].r1*100:.2f} & {stats['hloc-megaloc'].r5*100:.2f} & {stats['hloc-megaloc'].r10*100:.2f} \\",
        f"seqfg & {stats['seqfg'].num_queries:,} & {stats['seqfg'].mean:.3f} & {stats['seqfg'].median:.3f} & {stats['seqfg'].p90:.3f} & {stats['seqfg'].r1*100:.2f} & {stats['seqfg'].r5*100:.2f} & {stats['seqfg'].r10*100:.2f} \\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Top-1 Distance CDF: tỷ lệ truy vấn có khoảng cách top-1 $\leq \tau$ mét.}",
        r"\label{tab:ch4_core_top1_cdf}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Method & $\leq$1\,m & $\leq$2\,m & $\leq$3\,m & $\leq$5\,m & p90 (m) & p95 (m) \\",
        r"\hline",
        f"hloc-megaloc & {stats['hloc-megaloc'].cdf_1m*100:.2f} & {stats['hloc-megaloc'].cdf_2m*100:.2f} & {stats['hloc-megaloc'].cdf_3m*100:.2f} & {stats['hloc-megaloc'].cdf_5m*100:.2f} & {stats['hloc-megaloc'].p90:.3f} & {stats['hloc-megaloc'].p95:.3f} \\",
        f"seqfg & {stats['seqfg'].cdf_1m*100:.2f} & {stats['seqfg'].cdf_2m*100:.2f} & {stats['seqfg'].cdf_3m*100:.2f} & {stats['seqfg'].cdf_5m*100:.2f} & {stats['seqfg'].p90:.3f} & {stats['seqfg'].p95:.3f} \\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(pair_lines) + "\n", encoding="utf-8")


def make_appendix_tex(run_root: Path, out_path: Path) -> None:
    # Hard-coded illustrative cases are stable across the current run.
    success = CaseTrace(
        case_id="Case-Phone-S1",
        query_name="ios_2022-06-21_11.43.41_000/images/43466998493.jpg",
        baseline_top1=48.86168710199072,
        seqfg_top1=1.5678957667687994,
        baseline_hit=False,
        seqfg_hit=True,
        note="SeqFG sửa lỗi top-1 của baseline và kéo truy vấn về đúng vùng không gian.",
    )
    failure = CaseTrace(
        case_id="Case-Holo-F1",
        query_name="hl_2022-08-30-12-34-23-507.000/images/hetlf/744528595.jpg",
        baseline_top1=3.2048995319531883,
        seqfg_top1=96.38696040432554,
        baseline_hit=True,
        seqfg_hit=False,
        note="Đây là ca fragility cục bộ: baseline đúng nhưng SeqFG bị kéo lệch trong cùng scene HGE.",
    )
    lines = [
        r"% !TeX root = ../main.tex",
        "",
        r"\chapter{Ph\u1ee5 l\u1ee5c truy v\'et ca d\9nh t\9nh}\label{app:qual_case_trace}",
        "",
        r"Ph\u1ee5 l\u1ee5c này cung c\u1ea5p thông tin truy v\u1ebft cho hai ca điển hình được thảo luận trong Chương~\ref{chap:experiments}. Ca thành công thể hiện trường hợp SeqFG sửa được top-1 sai của baseline, còn ca thất bại cục bộ cho thấy SeqFG vẫn có thể bị kéo lệch trong điều kiện domain khó.",
        "",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Thông tin truy v\u1ebft cho hai ca định tính đại diện.}",
        r"  \label{tab:appendix_case_trace}",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \footnotesize",
        r"  \begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{1.6cm} >{\raggedright\arraybackslash}p{4.0cm} >{\raggedright\arraybackslash}p{4.0cm} >{\raggedright\arraybackslash}X}",
        r"    \hline",
        r"    Case ID & Truy vấn & Kết quả top-1 & Nhận xét truy v\u1ebft \\",
        r"    \hline",
        f"    {latex_escape(success.case_id)} & {latex_escape(success.query_name)} & HLOC: {success.baseline_top1:.2f} m ({'hit' if success.baseline_hit else 'miss'}); SeqFG: {success.seqfg_top1:.2f} m ({'hit' if success.seqfg_hit else 'miss'}) & {latex_escape(success.note)} \\",
        f"    {latex_escape(failure.case_id)} & {latex_escape(failure.query_name)} & HLOC: {failure.baseline_top1:.2f} m ({'hit' if failure.baseline_hit else 'miss'}); SeqFG: {failure.seqfg_top1:.2f} m ({'hit' if failure.seqfg_hit else 'miss'}) & {latex_escape(failure.note)} \\",
        r"    \hline",
        r"  \end{tabularx}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Chapter 4 thesis artifacts from benchmark outputs")
    p.add_argument("--summary-csv", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--thesis-root", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = parse_summary(args.summary_csv)
    evals = parse_eval_files(args.run_root)
    stats = compute_stats(evals)

    images_dir = args.thesis_root / "images"
    chapters_dir = args.thesis_root / "chapters"
    images_dir.mkdir(parents=True, exist_ok=True)
    chapters_dir.mkdir(parents=True, exist_ok=True)

    make_pairwise_recall_fig(rows, images_dir / "fig_ch4_recall_r1_by_pair.png")
    make_delta_fig(rows, images_dir / "fig_ch4_delta_r1_seqfg_minus_hloc_megaloc.png")
    make_cdf_fig(args.run_root, images_dir / "fig_ch4_top1_distance_cdf_core_methods.png")
    make_recall_profile_fig(stats, images_dir / "fig_ch4_recall_profile_core_methods.png")
    make_tables_tex(rows, stats, chapters_dir / "ch4_tables.tex")
    make_appendix_tex(args.run_root, chapters_dir / "appendix_case_trace.tex")

    print(chapters_dir / "ch4_tables.tex")
    print(chapters_dir / "appendix_case_trace.tex")
    for name in [
        "fig_ch4_recall_r1_by_pair.png",
        "fig_ch4_delta_r1_seqfg_minus_hloc_megaloc.png",
        "fig_ch4_top1_distance_cdf_core_methods.png",
        "fig_ch4_recall_profile_core_methods.png",
    ]:
        print(images_dir / name)


if __name__ == "__main__":
    main()
