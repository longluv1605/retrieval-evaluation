from __future__ import annotations

from copy import deepcopy
from collections import OrderedDict
import json
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
    def _build_sequence_groups(
        query_root: Path,
        query_names: list[str],
        sequence_cfg: dict[str, Any],
    ) -> tuple[OrderedDict[str, list[int]], dict[str, Any]]:
        sequence_file = str(sequence_cfg.get("sequence_file", "query_sequences.txt"))
        path = query_root / sequence_file

        if not path.exists():
            return OrderedDict({"__all__": list(range(len(query_names)))}), {
                "grouping_source": "single_global_group",
                "grouping_file": sequence_file,
                "num_sequences": 1,
                "num_unannotated": 0,
            }

        by_name: dict[str, str] = {}
        parse_errors = 0

        if path.suffix.lower() == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in sequence grouping file: {path}") from exc

            if isinstance(payload, list):
                # Supported JSON format: list of lists, each inner list is one sequence.
                for seq_idx, seq_images in enumerate(payload):
                    if not isinstance(seq_images, list):
                        parse_errors += 1
                        continue
                    sid = f"seq_{seq_idx:06d}"
                    for name in seq_images:
                        if not isinstance(name, str):
                            parse_errors += 1
                            continue
                        by_name[name] = sid
            elif isinstance(payload, dict):
                # Also support mapping sequence_id -> list[image_name].
                for sid, seq_images in payload.items():
                    if not isinstance(sid, str) or not isinstance(seq_images, list):
                        parse_errors += 1
                        continue
                    for name in seq_images:
                        if not isinstance(name, str):
                            parse_errors += 1
                            continue
                        by_name[name] = sid
            else:
                raise ValueError(
                    f"Unsupported JSON sequence grouping format in {path}. "
                    "Expected list[list[str]] or dict[str, list[str]]."
                )
        else:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    ln = line.strip()
                    if not ln or ln.startswith("#"):
                        continue
                    parts = ln.split()
                    if len(parts) < 2:
                        parse_errors += 1
                        continue
                    by_name[parts[0]] = parts[1]

        if not by_name:
            raise ValueError(
                f"Sequence grouping file exists but has no valid entries: {path}. "
                "Expected .txt format: 'image_name sequence_id' or JSON list/dict format."
            )

        groups: OrderedDict[str, list[int]] = OrderedDict()
        unannotated = 0
        for idx, qname in enumerate(query_names):
            sid = by_name.get(qname)
            if sid is None:
                sid = f"__single__{idx:08d}"
                unannotated += 1
            groups.setdefault(sid, []).append(idx)

        return groups, {
            "grouping_source": "query_sequences_file",
            "grouping_file": sequence_file,
            "grouping_parse_errors": parse_errors,
            "num_sequences": len(groups),
            "num_annotated": len(query_names) - unannotated,
            "num_unannotated": unannotated,
            "unannotated_policy": "singleton_sequence",
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
            sequence_cfg = dict(method_cfg.get("sequence", {}))
            groups, grouping_meta = PairSelection._build_sequence_groups(
                query_root=query_root,
                query_names=query_names,
                sequence_cfg=sequence_cfg,
            )

            ranked = {}
            sequence_summaries: list[dict[str, Any]] = []
            for sid, idxs in groups.items():
                idx_arr = np.array(idxs, dtype=np.int64)
                qnames_seq = [query_names[i] for i in idxs]
                ranked_seq, meta_seq = sequence_probabilistic_rank(
                    sim=sim[idx_arr, :],
                    map_names=map_names,
                    query_names=qnames_seq,
                    map_positions=map_positions,
                    query_positions=query_positions,
                    query_root=query_root,
                    sequence_cfg=sequence_cfg,
                    top_k=top_k,
                )
                for qname in qnames_seq:
                    ranked[qname] = ranked_seq[qname]

                sequence_summaries.append(
                    {
                        "sequence_id": sid,
                        "num_queries": len(qnames_seq),
                        "timestamp_source": meta_seq.get("timestamp_source", "unknown"),
                    }
                )

            sequence_meta = {
                **grouping_meta,
                "num_sequences_effective": len(groups),
                "sequence_summaries": sequence_summaries,
            }
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
