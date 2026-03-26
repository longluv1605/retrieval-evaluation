from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from retrieval_eval.io.dataset import DatasetLayout, get_map_names, get_query_names
from retrieval_eval.io.outputs import OutputLayout, ensure_dirs, write_json, write_stage_status
from retrieval_eval.io.pairs import write_pairs
from retrieval_eval.tasks.pair_selection import PairSelection


def _sha256_dict(payload: dict[str, Any]) -> str:
    dumped = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(dumped).hexdigest()


def run_retrieve(
    dataset_root: Path,
    outputs_root: Path,
    dataset_name: str,
    scene: str,
    map_dir: str,
    query_set: str,
    retrieval_method: str,
    run_id: str,
    config: dict[str, Any],
) -> Path:
    data_layout = DatasetLayout(
        dataset_root=dataset_root,
        dataset=dataset_name,
        scene=scene,
        query_set=query_set,
        map_dir=map_dir,
    )
    out_layout = OutputLayout(
        outputs_root=outputs_root,
        dataset=dataset_name,
        scene=scene,
        map_dir=map_dir,
        query_set=query_set,
        retrieval_method=retrieval_method,
        run_id=run_id,
    )
    ensure_dirs(out_layout)

    map_names = get_map_names(data_layout)
    query_names = get_query_names(data_layout)

    method_conf = dict(config.get("retrieval", {}))

    write_stage_status(
        out_layout.meta_dir / "stage_status.json",
        stage="retrieve",
        status="running",
        details={"method": retrieval_method, "run_id": run_id},
    )

    ranked, method_used_config = PairSelection.run(
        map_images_dir=data_layout.map_images_dir,
        query_images_dir=data_layout.query_images_dir,
        map_names=map_names,
        query_names=query_names,
        workdir=out_layout.retrieval_dir,
        method_name=retrieval_method,
        config=method_conf,
    )

    pairs_path = out_layout.retrieval_dir / "pairs.txt"
    write_pairs(pairs_path, ranked)

    retrieval_cfg = {
        "config_path": str(config.get("_config_path", "")),
        "config_hash": _sha256_dict(method_used_config),
        "method": retrieval_method,
        "params": method_used_config,
    }
    write_json(out_layout.meta_dir / "retrieval_config.json", retrieval_cfg)

    write_stage_status(
        out_layout.meta_dir / "stage_status.json",
        stage="retrieve",
        status="done",
        details={
            "method": retrieval_method,
            "run_id": run_id,
            "num_map": len(map_names),
            "num_query": len(query_names),
            "pairs_path": str(pairs_path),
        },
    )

    return pairs_path
