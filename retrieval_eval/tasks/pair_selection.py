from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .feature_extraction import RetrievalFeatureExtraction


def _read_global_descriptors(feature_h5: Path, names: list[str]) -> np.ndarray:
    desc = []
    with h5py.File(feature_h5, "r") as fd:
        for name in names:
            if name not in fd:
                raise KeyError(f"Missing descriptor for image: {name}")
            grp = fd[name]
            if "global_descriptor" not in grp:
                raise KeyError(f"Image {name} has no global_descriptor in {feature_h5}")
            vec = np.asarray(grp["global_descriptor"], dtype=np.float32).reshape(-1)
            desc.append(vec)
    arr = np.stack(desc, axis=0)
    arr = arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), a_min=1e-8, a_max=None)
    return arr


class PairSelection:
    """LAMAR-style retrieval pair selection task."""

    methods: dict[str, dict[str, Any]] = {
        "hloc-netvlad": {
            "name": "hloc-netvlad",
            "feature": RetrievalFeatureExtraction.methods["netvlad"],
        },
        "hloc-megaloc": {
            "name": "hloc-megaloc",
            "feature": RetrievalFeatureExtraction.methods["megaloc"],
        },
        "hloc-dir": {
            "name": "hloc-dir",
            "feature": RetrievalFeatureExtraction.methods["dir"],
        },
    }

    @staticmethod
    def run(
        map_images_dir: Path,
        query_images_dir: Path,
        map_names: list[str],
        query_names: list[str],
        workdir: Path,
        method_name: str,
        config: dict[str, Any],
    ) -> tuple[dict[str, list[tuple[str, float]]], dict[str, Any]]:
        if method_name not in PairSelection.methods:
            raise ValueError(f"Unknown retrieval method: {method_name}")

        method = deepcopy(PairSelection.methods[method_name])
        method_cfg = deepcopy(config)
        top_k = int(method_cfg.get("top_k", 20))
        overwrite = bool(method_cfg.get("overwrite_features", False))

        workdir.mkdir(parents=True, exist_ok=True)
        map_h5 = workdir / "map_features.h5"
        query_h5 = workdir / "query_features.h5"

        RetrievalFeatureExtraction.extract(
            image_dir=map_images_dir,
            image_list=map_names,
            feature_path=map_h5,
            method_conf=method["feature"],
            overwrite=overwrite,
        )
        RetrievalFeatureExtraction.extract(
            image_dir=query_images_dir,
            image_list=query_names,
            feature_path=query_h5,
            method_conf=method["feature"],
            overwrite=overwrite,
        )

        map_desc = _read_global_descriptors(map_h5, map_names)
        query_desc = _read_global_descriptors(query_h5, query_names)
        sim = np.matmul(query_desc, map_desc.T)

        k_eff = min(top_k, len(map_names))
        idx = np.argsort(-sim, axis=1)[:, :k_eff]

        ranked: dict[str, list[tuple[str, float]]] = {}
        for qi, qname in enumerate(query_names):
            ranked[qname] = [(map_names[mi], float(sim[qi, mi])) for mi in idx[qi]]

        retrieval_config = {
            "method": method_name,
            "feature": method["feature"]["name"],
            "top_k": top_k,
            "overwrite_features": overwrite,
        }
        return ranked, retrieval_config
