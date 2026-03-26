from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from retrieval_eval.io.poses import load_positions
from retrieval_eval.methods.sequence_prob import sequence_probabilistic_rank

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
        "seqprob": {
            "name": "seqprob",
            "sequence": True,
        },
        "seqfg": {
            "name": "seqfg",
            "sequence": True,
        },
    }

    @staticmethod
    def _resolve_feature(method_name: str, method_cfg: dict[str, Any]) -> dict[str, Any]:
        if method_name in {"seqprob", "seqfg"}:
            global_model = str(method_cfg.get("global_model", "megaloc"))
            if global_model not in RetrievalFeatureExtraction.methods:
                supported = ", ".join(sorted(RetrievalFeatureExtraction.methods))
                raise ValueError(
                    f"Unknown retrieval.global_model={global_model!r}. Supported values: {supported}"
                )
            return RetrievalFeatureExtraction.methods[global_model]
        return PairSelection.methods[method_name]["feature"]

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

        method_cfg = deepcopy(config)
        top_k = int(method_cfg.get("top_k", 20))
        overwrite = bool(method_cfg.get("overwrite_features", False))
        feature_conf = PairSelection._resolve_feature(method_name, method_cfg)

        workdir.mkdir(parents=True, exist_ok=True)
        map_h5 = workdir / "map_features.h5"
        query_h5 = workdir / "query_features.h5"

        RetrievalFeatureExtraction.extract(
            image_dir=map_images_dir,
            image_list=map_names,
            feature_path=map_h5,
            method_conf=feature_conf,
            overwrite=overwrite,
        )
        RetrievalFeatureExtraction.extract(
            image_dir=query_images_dir,
            image_list=query_names,
            feature_path=query_h5,
            method_conf=feature_conf,
            overwrite=overwrite,
        )

        map_desc = _read_global_descriptors(map_h5, map_names)
        query_desc = _read_global_descriptors(query_h5, query_names)
        sim = np.matmul(query_desc, map_desc.T)

        sequence_meta: dict[str, Any] = {}
        if method_name in {"seqprob", "seqfg"}:
            map_root = map_images_dir.parent
            query_root = query_images_dir.parent
            map_positions = load_positions(map_root / "poses.txt")
            if not map_positions:
                raise FileNotFoundError(
                    f"Sequence method requires map poses, but none were loaded from: {map_root / 'poses.txt'}"
                )
            missing_map_pose = [name for name in map_names if name not in map_positions]
            if missing_map_pose:
                preview = ", ".join(missing_map_pose[:5])
                raise ValueError(
                    "Sequence method requires map poses for all map images. "
                    f"Missing {len(missing_map_pose)} poses (examples: {preview})"
                )
            query_positions = load_positions(query_root / "poses.txt")
            ranked, sequence_meta = sequence_probabilistic_rank(
                sim=sim,
                map_names=map_names,
                query_names=query_names,
                map_positions=map_positions,
                query_positions=query_positions,
                query_root=query_root,
                sequence_cfg=dict(method_cfg.get("sequence", {})),
                top_k=top_k,
            )
        else:
            k_eff = min(top_k, len(map_names))
            idx = np.argsort(-sim, axis=1)[:, :k_eff]

            ranked = {}
            for qi, qname in enumerate(query_names):
                ranked[qname] = [(map_names[mi], float(sim[qi, mi])) for mi in idx[qi]]

        retrieval_config = {
            "method": method_name,
            "feature": feature_conf["name"],
            "top_k": top_k,
            "overwrite_features": overwrite,
        }
        if sequence_meta:
            retrieval_config["sequence"] = sequence_meta
        return ranked, retrieval_config
