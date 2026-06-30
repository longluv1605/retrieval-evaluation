from __future__ import annotations

from collections import defaultdict
from pathlib import Path


def write_pairs(path: Path, ranked_pairs: dict[str, list[tuple[str, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for qname, rows in ranked_pairs.items():
            for rank, (mname, score) in enumerate(rows, start=1):
                f.write(f"{qname} {mname} {score:.8f} {rank}\n")


def read_pairs(path: Path) -> dict[str, list[tuple[str, float, int]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing pairs file: {path}")

    pairs: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ln = line.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) < 4:
                continue
            qname, mname = parts[0], parts[1]
            score, rank = float(parts[2]), int(parts[3])
            pairs[qname].append((mname, score, rank))

    for qname in pairs:
        pairs[qname].sort(key=lambda x: x[2])
    return dict(pairs)
