from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def parse_timestamp_unit_to_scale(unit: str | None) -> float:
    """Return multiplicative scale to convert raw timestamp into seconds."""
    normalized = (unit or "us").strip().lower()
    scales = {
        "s": 1.0,
        "ms": 1e-3,
        "us": 1e-6,
        "ns": 1e-9,
    }
    if normalized not in scales:
        allowed = ", ".join(sorted(scales))
        raise ValueError(f"Invalid timestamp_unit={unit!r}. Allowed values: {allowed}")
    return scales[normalized]


def load_query_timestamps_seconds(
    query_root: Path,
    query_names: list[str],
    timestamp_unit: str,
    dt_s_default: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load query timestamps and convert to seconds.

    Fallback policy:
    - Missing timestamps.txt -> uniform timeline.
    - Missing key for any query or parse failure -> uniform timeline.
    """
    step_s = float(dt_s_default)
    if step_s <= 0:
        raise ValueError("retrieval.sequence.dt_s_default must be > 0")

    fallback_ts = np.arange(len(query_names), dtype=np.float64) * step_s
    path = query_root / "timestamps.txt"
    if not path.exists():
        return fallback_ts, {
            "timestamp_source": "fallback_uniform",
            "timestamp_fallback_reason": "missing_file",
            "timestamp_unit": timestamp_unit,
            "dt_s_default": step_s,
        }

    scale = parse_timestamp_unit_to_scale(timestamp_unit)
    ts_raw: dict[str, float] = {}
    parse_errors = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ln = line.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) < 2:
                parse_errors += 1
                continue
            name = parts[0]
            try:
                ts_raw[name] = float(parts[1])
            except ValueError:
                parse_errors += 1

    missing = [name for name in query_names if name not in ts_raw]
    if parse_errors > 0 or missing:
        reason = "parse_errors" if parse_errors > 0 else "missing_keys"
        return fallback_ts, {
            "timestamp_source": "fallback_uniform",
            "timestamp_fallback_reason": reason,
            "timestamp_missing_keys": len(missing),
            "timestamp_parse_errors": parse_errors,
            "timestamp_unit": timestamp_unit,
            "dt_s_default": step_s,
        }

    ts_s = np.array([ts_raw[name] * scale for name in query_names], dtype=np.float64)
    return ts_s, {
        "timestamp_source": "timestamps_txt",
        "timestamp_unit": timestamp_unit,
        "dt_s_default": step_s,
    }


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    stable = x - x_max
    out = x_max + np.log(np.sum(np.exp(stable), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def _candidate_indices(sim: np.ndarray, num_candidates: int) -> tuple[np.ndarray, np.ndarray]:
    q_count, map_count = sim.shape
    c_eff = min(max(1, int(num_candidates)), map_count)
    idx = np.argpartition(-sim, kth=c_eff - 1, axis=1)[:, :c_eff]
    cand_scores = np.take_along_axis(sim, idx, axis=1)
    order = np.argsort(-cand_scores, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    cand_scores = np.take_along_axis(cand_scores, order, axis=1)
    assert idx.shape == (q_count, c_eff)
    assert cand_scores.shape == (q_count, c_eff)
    return idx, cand_scores


def temporal_smooth_rank(
    sim: np.ndarray,
    map_names: list[str],
    query_names: list[str],
    sequence_cfg: dict[str, Any],
    top_k: int,
) -> tuple[dict[str, list[tuple[str, float]]], dict[str, Any]]:
    """Rank map images after temporal moving-average smoothing of similarity rows."""
    if sim.ndim != 2:
        raise ValueError(f"sim must be 2D, got shape={sim.shape}")

    q_count, map_count = sim.shape
    if q_count != len(query_names):
        raise ValueError("sim/query_names size mismatch")
    if map_count != len(map_names):
        raise ValueError("sim/map_names size mismatch")

    window_radius = max(0, int(sequence_cfg.get("smooth_window_radius", 2)))
    blend_weight = float(sequence_cfg.get("smooth_weight", 1.0))
    blend_weight = float(np.clip(blend_weight, 0.0, 1.0))

    smoothed = np.empty_like(sim, dtype=np.float64)
    sim64 = sim.astype(np.float64, copy=False)
    for i in range(q_count):
        lo = max(0, i - window_radius)
        hi = min(q_count, i + window_radius + 1)
        smoothed[i] = np.mean(sim64[lo:hi], axis=0)

    final_scores = (1.0 - blend_weight) * sim64 + blend_weight * smoothed
    k_eff = min(max(1, int(top_k)), map_count)
    rank_idx = np.argsort(-final_scores, axis=1)[:, :k_eff]

    ranked: dict[str, list[tuple[str, float]]] = {}
    for qi, qname in enumerate(query_names):
        ranked[qname] = [(map_names[int(mi)], float(final_scores[qi, mi])) for mi in rank_idx[qi]]

    return ranked, {
        "smooth_window_radius": window_radius,
        "smooth_weight": blend_weight,
        "num_queries": q_count,
        "num_map": map_count,
    }


def _candidate_positions(
    cand_idx: np.ndarray,
    map_names: list[str],
    map_positions: dict[str, np.ndarray],
) -> np.ndarray:
    q_count, c_count = cand_idx.shape
    out = np.full((q_count, c_count, 3), np.nan, dtype=np.float64)
    for qi in range(q_count):
        for ci in range(c_count):
            pos = map_positions.get(map_names[int(cand_idx[qi, ci])])
            if pos is None:
                continue
            out[qi, ci] = np.asarray(pos, dtype=np.float64)
    return out


def _query_step_distances(query_names: list[str], query_positions: dict[str, np.ndarray]) -> np.ndarray:
    d = np.full(len(query_names), np.nan, dtype=np.float64)
    for i in range(1, len(query_names)):
        p0 = query_positions.get(query_names[i - 1])
        p1 = query_positions.get(query_names[i])
        if p0 is None or p1 is None:
            continue
        d[i] = float(np.linalg.norm(np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64)))
    return d


def _build_transition_log_probs(
    cand_pos: np.ndarray,
    dt_s: np.ndarray,
    query_step_d: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    q_count, c_count, _ = cand_pos.shape
    out = np.zeros((q_count - 1, c_count, c_count), dtype=np.float64)

    v_max = float(cfg["v_max"])
    eps = max(float(cfg["eps"]), 1e-9)
    sigma_d = max(float(cfg["sigma_d"]), eps)
    lambda_dist = max(float(cfg["lambda_dist"]), 0.0)
    transition_weight = max(float(cfg.get("transition_weight", 1.0)), 0.0)
    max_dt_s = max(float(cfg["max_dt_s"]), eps)

    for i in range(1, q_count):
        dt = float(dt_s[i] - dt_s[i - 1])
        if not np.isfinite(dt):
            dt = float(cfg["dt_s_default"])
        dt = min(max(dt, eps), max_dt_s)

        prev_pos = cand_pos[i - 1]  # [C, 3]
        cur_pos = cand_pos[i]  # [C, 3]
        prev_ok = np.all(np.isfinite(prev_pos), axis=1)
        cur_ok = np.all(np.isfinite(cur_pos), axis=1)
        if not np.any(prev_ok) or not np.any(cur_ok):
            continue

        diff = prev_pos[:, None, :] - cur_pos[None, :, :]
        map_dist = np.linalg.norm(diff, axis=2)

        expected_dist = v_max * dt
        if np.isfinite(query_step_d[i]) and lambda_dist > 0:
            expected_dist = (1.0 - np.clip(lambda_dist, 0.0, 1.0)) * expected_dist + np.clip(
                lambda_dist, 0.0, 1.0
            ) * float(query_step_d[i])

        sigma = max(sigma_d * dt, eps)
        transition = -0.5 * ((map_dist - expected_dist) / sigma) ** 2

        if v_max > 0:
            over = np.maximum(map_dist - v_max * dt, 0.0)
            transition -= (over / sigma) ** 2

        transition *= transition_weight

        valid = prev_ok[:, None] & cur_ok[None, :]
        transition[~valid] = 0.0
        out[i - 1] = transition

    return out


def _fuse_sequence_scores(
    cand_scores: np.ndarray,
    post_prob: np.ndarray,
    log_gamma: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    fusion = str(cfg.get("fusion", "prob_add")).strip().lower()
    weight = float(cfg.get("fusion_weight", 1.0))

    if fusion == "prob_add":
        return cand_scores + weight * post_prob
    if fusion == "logprob_add":
        return cand_scores + weight * log_gamma
    if fusion == "prob_only":
        return post_prob
    if fusion == "logprob_only":
        return log_gamma

    allowed = "logprob_add, logprob_only, prob_add, prob_only"
    raise ValueError(f"Invalid retrieval.sequence.fusion={fusion!r}. Allowed values: {allowed}")


def sequence_probabilistic_rank(
    sim: np.ndarray,
    map_names: list[str],
    query_names: list[str],
    map_positions: dict[str, np.ndarray],
    query_positions: dict[str, np.ndarray],
    query_root: Path,
    sequence_cfg: dict[str, Any],
    top_k: int,
) -> tuple[dict[str, list[tuple[str, float]]], dict[str, Any]]:
    """Rank map images with sequence probabilistic retrieval (forward-backward marginals)."""
    if sim.ndim != 2:
        raise ValueError(f"sim must be 2D, got shape={sim.shape}")

    seq_cfg = {
        "num_candidates": int(sequence_cfg.get("num_candidates", 50)),
        "tau": float(sequence_cfg.get("tau", 0.07)),
        "v_max": float(sequence_cfg.get("v_max", 3.0)),
        "eps": float(sequence_cfg.get("eps", 0.25)),
        "use_query_displacement": bool(sequence_cfg.get("use_query_displacement", True)),
        "lambda_dist": float(sequence_cfg.get("lambda_dist", 1.0)),
        "sigma_d": float(sequence_cfg.get("sigma_d", 1.0)),
        "transition_weight": max(float(sequence_cfg.get("transition_weight", 1.0)), 0.0),
        "fusion": str(sequence_cfg.get("fusion", "prob_add")),
        "fusion_weight": float(sequence_cfg.get("fusion_weight", 1.0)),
        "max_dt_s": float(sequence_cfg.get("max_dt_s", 30.0)),
        "timestamp_unit": str(sequence_cfg.get("timestamp_unit", "us")),
        "dt_s_default": float(sequence_cfg.get("dt_s_default", 10.0)),
    }

    q_count, map_count = sim.shape
    if q_count != len(query_names):
        raise ValueError("sim/query_names size mismatch")
    if map_count != len(map_names):
        raise ValueError("sim/map_names size mismatch")

    timestamps_s, ts_meta = load_query_timestamps_seconds(
        query_root=query_root,
        query_names=query_names,
        timestamp_unit=seq_cfg["timestamp_unit"],
        dt_s_default=seq_cfg["dt_s_default"],
    )

    cand_idx, cand_scores = _candidate_indices(sim, seq_cfg["num_candidates"])
    tau = max(seq_cfg["tau"], 1e-8)
    log_emission = cand_scores / tau
    log_emission -= _logsumexp(log_emission, axis=1)[:, None]

    cand_pos = _candidate_positions(cand_idx=cand_idx, map_names=map_names, map_positions=map_positions)
    if seq_cfg["use_query_displacement"]:
        query_step_d = _query_step_distances(query_names=query_names, query_positions=query_positions)
    else:
        query_step_d = np.full(q_count, np.nan, dtype=np.float64)

    if q_count <= 1:
        log_gamma = log_emission
    else:
        log_trans = _build_transition_log_probs(
            cand_pos=cand_pos,
            dt_s=timestamps_s,
            query_step_d=query_step_d,
            cfg=seq_cfg,
        )

        c_count = cand_idx.shape[1]
        alpha = np.zeros((q_count, c_count), dtype=np.float64)
        beta = np.zeros((q_count, c_count), dtype=np.float64)
        alpha[0] = log_emission[0]
        for i in range(1, q_count):
            trans_scores = alpha[i - 1][:, None] + log_trans[i - 1]
            alpha[i] = log_emission[i] + _logsumexp(trans_scores, axis=0)

        for i in range(q_count - 2, -1, -1):
            trans_scores = log_trans[i] + log_emission[i + 1][None, :] + beta[i + 1][None, :]
            beta[i] = _logsumexp(trans_scores, axis=1)
        log_gamma = alpha + beta

    log_gamma -= _logsumexp(log_gamma, axis=1)[:, None]
    post_prob = np.exp(log_gamma)

    # Keep dense output for compatibility with existing pairs.txt writer.
    seq_scores = sim.astype(np.float64, copy=True)
    rows = np.arange(q_count)[:, None]
    seq_scores[rows, cand_idx] = _fuse_sequence_scores(
        cand_scores=cand_scores,
        post_prob=post_prob,
        log_gamma=log_gamma,
        cfg=seq_cfg,
    )

    k_eff = min(max(1, int(top_k)), map_count)
    rank_idx = np.argsort(-seq_scores, axis=1)[:, :k_eff]
    ranked: dict[str, list[tuple[str, float]]] = {}
    for qi, qname in enumerate(query_names):
        ranked[qname] = [(map_names[int(mi)], float(seq_scores[qi, mi])) for mi in rank_idx[qi]]

    used_query_poses = sum(1 for qname in query_names if qname in query_positions)
    used_map_poses = len(map_positions)
    meta = {
        **seq_cfg,
        **ts_meta,
        "num_candidates_effective": int(cand_idx.shape[1]),
        "num_query_with_pose": int(used_query_poses),
        "num_map_with_pose": int(used_map_poses),
    }
    if not seq_cfg["use_query_displacement"]:
        meta["lambda_dist_effective"] = 0.0
        meta["sequence_note"] = "query displacement disabled; motion prior uses timestamps and v_max only"
    elif used_query_poses == 0:
        meta["lambda_dist_effective"] = 0.0
        meta["sequence_note"] = "query poses missing, motion prior uses v_max only"
    else:
        meta["lambda_dist_effective"] = float(np.clip(seq_cfg["lambda_dist"], 0.0, 1.0))
    return ranked, meta
