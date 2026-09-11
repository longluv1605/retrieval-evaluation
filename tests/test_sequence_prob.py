from __future__ import annotations

import numpy as np

from retrieval_eval.methods.sequence_prob import sequence_probabilistic_rank


def _write_timestamps(root, names):
    root.mkdir(parents=True, exist_ok=True)
    (root / "timestamps.txt").write_text(
        "\n".join(f"{name} {idx}" for idx, name in enumerate(names)) + "\n",
        encoding="utf-8",
    )


def _base_cfg(**overrides):
    cfg = {
        "num_candidates": 4,
        "tau": 0.01,
        "v_max": 1.0,
        "eps": 0.1,
        "use_query_displacement": False,
        "lambda_dist": 0.0,
        "sigma_d": 1.0,
        "transition_weight": 1.0,
        "fusion": "logprob_add",
        "fusion_weight": 0.05,
        "max_dt_s": 30.0,
        "timestamp_unit": "s",
        "dt_s_default": 1.0,
    }
    cfg.update(overrides)
    return cfg


def test_timestamp_only_does_not_depend_on_query_poses(tmp_path):
    query_names = ["q0.jpg", "q1.jpg", "q2.jpg"]
    map_names = ["m0.jpg", "m1.jpg", "m2.jpg", "alias.jpg"]
    _write_timestamps(tmp_path, query_names)
    sim = np.array(
        [
            [1.00, 0.20, 0.10, 0.30],
            [0.20, 0.98, 0.20, 0.99],
            [0.10, 0.20, 1.00, 0.30],
        ],
        dtype=np.float64,
    )
    map_positions = {
        "m0.jpg": np.array([0.0, 0.0, 0.0]),
        "m1.jpg": np.array([1.0, 0.0, 0.0]),
        "m2.jpg": np.array([2.0, 0.0, 0.0]),
        "alias.jpg": np.array([100.0, 0.0, 0.0]),
    }
    query_positions = {name: np.array([1000.0 * idx, 0.0, 0.0]) for idx, name in enumerate(query_names)}

    no_pose, _ = sequence_probabilistic_rank(
        sim, map_names, query_names, map_positions, {}, tmp_path, _base_cfg(), top_k=3
    )
    with_pose, meta = sequence_probabilistic_rank(
        sim, map_names, query_names, map_positions, query_positions, tmp_path, _base_cfg(), top_k=3
    )

    assert no_pose == with_pose
    assert meta["lambda_dist_effective"] == 0.0


def test_temporal_factor_promotes_map_consistent_alias_correction(tmp_path):
    query_names = ["q0.jpg", "q1.jpg", "q2.jpg"]
    map_names = ["m0.jpg", "m1.jpg", "m2.jpg", "alias.jpg"]
    _write_timestamps(tmp_path, query_names)
    sim = np.array(
        [
            [1.00, 0.20, 0.10, 0.30],
            [0.20, 0.98, 0.20, 0.99],
            [0.10, 0.20, 1.00, 0.30],
        ],
        dtype=np.float64,
    )
    map_positions = {
        "m0.jpg": np.array([0.0, 0.0, 0.0]),
        "m1.jpg": np.array([1.0, 0.0, 0.0]),
        "m2.jpg": np.array([2.0, 0.0, 0.0]),
        "alias.jpg": np.array([100.0, 0.0, 0.0]),
    }

    ranked, _ = sequence_probabilistic_rank(
        sim,
        map_names,
        query_names,
        map_positions,
        {},
        tmp_path,
        _base_cfg(fusion_weight=0.20),
        top_k=1,
    )

    assert ranked["q1.jpg"][0][0] == "m1.jpg"


def test_unary_only_preserves_independent_order(tmp_path):
    query_names = ["q0.jpg", "q1.jpg"]
    map_names = ["m0.jpg", "m1.jpg", "m2.jpg"]
    _write_timestamps(tmp_path, query_names)
    sim = np.array([[0.9, 0.8, 0.1], [0.2, 0.7, 0.6]], dtype=np.float64)
    map_positions = {
        "m0.jpg": np.array([0.0, 0.0, 0.0]),
        "m1.jpg": np.array([1.0, 0.0, 0.0]),
        "m2.jpg": np.array([2.0, 0.0, 0.0]),
    }

    ranked, _ = sequence_probabilistic_rank(
        sim,
        map_names,
        query_names,
        map_positions,
        {},
        tmp_path,
        _base_cfg(transition_weight=0.0),
        top_k=3,
    )

    assert [name for name, _score in ranked["q0.jpg"]] == ["m0.jpg", "m1.jpg", "m2.jpg"]
    assert [name for name, _score in ranked["q1.jpg"]] == ["m1.jpg", "m2.jpg", "m0.jpg"]


def test_published_fusion_adds_marginal_probability_to_similarity(tmp_path):
    """prob_add with alpha=1 must reproduce eq. s_i(j) = S_ij + alpha * P_i(k).

    Guards the paper's commensurate-by-construction argument: the fused term is a
    marginal probability in [0, 1], not a log-probability.
    """
    query_names = ["q0.jpg", "q1.jpg", "q2.jpg"]
    map_names = ["m0.jpg", "m1.jpg", "m2.jpg", "alias.jpg"]
    _write_timestamps(tmp_path, query_names)
    sim = np.array(
        [
            [1.00, 0.20, 0.10, 0.30],
            [0.20, 0.98, 0.20, 0.99],
            [0.10, 0.20, 1.00, 0.30],
        ],
        dtype=np.float64,
    )
    map_positions = {
        "m0.jpg": np.array([0.0, 0.0, 0.0]),
        "m1.jpg": np.array([1.0, 0.0, 0.0]),
        "m2.jpg": np.array([2.0, 0.0, 0.0]),
        "alias.jpg": np.array([100.0, 0.0, 0.0]),
    }
    cfg = _base_cfg(fusion="prob_add", fusion_weight=1.0, num_candidates=4)

    ranked, _ = sequence_probabilistic_rank(
        sim, map_names, query_names, map_positions, {}, tmp_path, cfg, top_k=4
    )

    sim_lookup = {
        (qi, name): float(sim[qi, mi]) for mi, name in enumerate(map_names) for qi in range(len(query_names))
    }
    for qi in range(len(query_names)):
        scores = dict(ranked[query_names[qi]])
        # Every candidate score is its similarity plus a value in [0, 1], and the
        # increments over one frame form a distribution summing to 1.
        increments = [scores[name] - sim_lookup[(qi, name)] for name in scores]
        assert all(-1e-9 <= inc <= 1.0 + 1e-9 for inc in increments), increments
        assert abs(sum(increments) - 1.0) < 1e-6, sum(increments)


def test_stationary_camera_is_not_penalized_by_speed_limit_term():
    """The one-sided hinge (term b of eq. psi) must vanish for zero displacement.

    Reviewer concern R8.5: V_max * dt is a speed bound, not a displacement
    estimate, so exceeding it is penalized but staying under it is not.
    """
    from retrieval_eval.methods.sequence_prob import _build_transition_log_probs

    cfg = {
        "v_max": 3.0,
        "eps": 0.1,
        "sigma_d": 1.0,
        "lambda_dist": 0.0,
        "transition_weight": 1.0,
        "max_dt_s": 30.0,
        "dt_s_default": 1.0,
    }
    # Two frames, two candidates: co-located (no displacement) vs far beyond v_max*dt.
    cand_pos = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    dt_s = np.array([0.0, 1.0], dtype=np.float64)
    query_step_d = np.array([np.nan, np.nan], dtype=np.float64)

    log_trans = _build_transition_log_probs(cand_pos, dt_s, query_step_d, cfg)

    stationary = log_trans[0, 0, 0]
    over_limit = log_trans[0, 0, 1]
    # Staying put costs only the two-sided term; the hinge contributes nothing.
    expected_two_sided = -0.5 * ((0.0 - cfg["v_max"] * 1.0) / 1.0) ** 2
    assert abs(stationary - expected_two_sided) < 1e-9
    # Exceeding the bound is penalized far more heavily.
    assert over_limit < stationary - 100.0
