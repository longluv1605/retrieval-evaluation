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
