from __future__ import annotations

from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Any


class RetrievalFeatureExtraction:
    """Global feature extraction wrapper (LAMAR-style methods registry)."""

    methods: dict[str, dict[str, Any]] = {
        "netvlad": {
            "name": "netvlad",
            "hloc": {
                "model": {"name": "netvlad"},
                "preprocessing": {"resize_max": 640},
            },
        },
        "megaloc": {
            "name": "megaloc",
            "hloc": {
                "model": {"name": "megaloc"},
                "preprocessing": {"resize_max": 640},
            },
        },
        "dir": {
            "name": "dir",
            "hloc": {
                "model": {"name": "dir"},
                "preprocessing": {"resize_max": 640},
            },
        },
    }

    @staticmethod
    def extract(
        image_dir: Path,
        image_list: list[str],
        feature_path: Path,
        method_conf: dict[str, Any],
        overwrite: bool = False,
    ) -> None:
        # Lazy import to keep package importable without hloc installed.
        try:
            extract_features = import_module("hloc.extract_features")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "hloc is required for retrieval feature extraction. "
                "Install it with: pip install -e ./hloc"
            ) from exc

        conf = deepcopy(method_conf.get("hloc", {}))
        if "output" not in conf:
            conf["output"] = method_conf.get("name", "global")

        extract_features.main(
            conf,
            image_dir=image_dir,
            image_list=image_list,
            feature_path=feature_path,
            overwrite=overwrite,
        )
