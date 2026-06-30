from __future__ import annotations

from pathlib import Path

import numpy as np


def load_positions(poses_path: Path) -> dict[str, np.ndarray]:
    """Load camera centers from poses.txt format.

    Line format: image_name tx ty tz qx qy qz qw
    """
    if not poses_path.exists():
        return {}

    positions: dict[str, np.ndarray] = {}
    with poses_path.open("r", encoding="utf-8") as f:
        for line in f:
            ln = line.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) < 8:
                continue
            name = parts[0]
            tx, ty, tz = map(float, parts[1:4])
            positions[name] = np.array([tx, ty, tz], dtype=np.float64)
    return positions
