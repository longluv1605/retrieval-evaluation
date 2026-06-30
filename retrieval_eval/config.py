from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


SEQUENCE_DEFAULTS: dict[str, Any] = {
    "sequence_file": "query_sequences.txt",
    "num_candidates": 100,
    "tau": 0.01,
    "v_max": 3.0,
    "eps": 0.1,
    "use_query_displacement": False,
    "lambda_dist": 0.0,
    "sigma_d": 1.0,
    "transition_weight": 1.0,
    "fusion": "logprob_add",
    "fusion_weight": 0.05,
    "max_dt_s": 30.0,
    "timestamp_unit": "us",
    "dt_s_default": 10.0,
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return data


def normalize_retrieval_config(config: dict[str, Any]) -> dict[str, Any]:
    """Apply retrieval schema defaults, including sequence settings."""
    retrieval = dict(config.get("retrieval", {}))
    sequence = dict(SEQUENCE_DEFAULTS)
    sequence.update(dict(retrieval.get("sequence", {})))
    retrieval["sequence"] = sequence

    retrieval.setdefault("top_k", 20)
    retrieval.setdefault("overwrite_features", False)
    retrieval.setdefault("global_model", "megaloc")
    return retrieval
