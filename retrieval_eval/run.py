from __future__ import annotations

from pathlib import Path
from typing import Any

from retrieval_eval.tasks.eval_distance import run_eval_distance
from retrieval_eval.tasks.retrieve import run_retrieve


def run(
    dataset_root: Path,
    outputs_root: Path,
    dataset_name: str,
    scene: str,
    map_dir: str,
    query_set: str,
    retrieval_method: str,
    run_id: str,
    retrieval_config: dict[str, Any],
    eval_config: dict[str, Any],
    threshold_m: float,
) -> tuple[Path, Path]:
    """Run retrieval then distance evaluation."""
    pairs_path = run_retrieve(
        dataset_root=dataset_root,
        outputs_root=outputs_root,
        dataset_name=dataset_name,
        scene=scene,
        map_dir=map_dir,
        query_set=query_set,
        retrieval_method=retrieval_method,
        run_id=run_id,
        config=retrieval_config,
    )
    eval_path = run_eval_distance(
        dataset_root=dataset_root,
        outputs_root=outputs_root,
        dataset_name=dataset_name,
        scene=scene,
        map_dir=map_dir,
        query_set=query_set,
        retrieval_method=retrieval_method,
        run_id=run_id,
        config=eval_config,
        threshold_m=threshold_m,
    )
    return pairs_path, eval_path
