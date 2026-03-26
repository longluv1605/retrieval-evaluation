from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Pose:
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float


Key = tuple[str, str]  # (timestamp, sensor_or_device)


def _parse_csv_like_line(line: str) -> list[str]:
    return [p.strip() for p in line.split(",")]


def _read_images_txt(path: Path) -> dict[Key, str]:
    rows: dict[Key, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = _parse_csv_like_line(line)
            if len(parts) < 3:
                continue
            timestamp, sensor_id, image_path = parts[0], parts[1], parts[2]
            rows[(timestamp, sensor_id)] = image_path
    return rows


def _read_trajectories(path: Path) -> dict[Key, Pose]:
    rows: dict[Key, Pose] = {}
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = _parse_csv_like_line(line)
            if len(parts) < 9:
                continue
            timestamp, device_id = parts[0], parts[1]
            qw, qx, qy, qz = map(float, parts[2:6])
            tx, ty, tz = map(float, parts[6:9])
            rows[(timestamp, device_id)] = Pose(qw=qw, qx=qx, qy=qy, qz=qz, tx=tx, ty=ty, tz=tz)
    return rows


def _read_queries(path: Path) -> list[Key]:
    if not path.exists():
        return []

    rows: list[Key] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = _parse_csv_like_line(line)
            if len(parts) < 2:
                continue
            rows.append((parts[0], parts[1]))
    return rows


def _resolve_query_image(
    query_key: Key,
    images: dict[Key, str],
    sensor_priority: list[str],
) -> tuple[str | None, Key | None]:
    if query_key in images:
        return images[query_key], query_key

    ts, _device = query_key
    same_ts = [(k, v) for k, v in images.items() if k[0] == ts]
    if not same_ts:
        return None, None

    for hint in sensor_priority:
        for key, image_path in same_ts:
            if hint in key[1]:
                return image_path, key

    same_ts.sort(key=lambda kv: kv[0][1])
    key, image_path = same_ts[0]
    return image_path, key


def _safe_link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_images:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def _normalize_session_token(token: str) -> list[str]:
    candidates = [token]
    if token.endswith("_000"):
        candidates.append(token[: -len("_000")])
    if token.endswith(".000"):
        candidates.append(token[: -len(".000")])
    # Keep order stable and remove duplicates.
    return list(dict.fromkeys(candidates))


def _resolve_image_source(session_src: Path, image_path: str) -> Path | None:
    """Resolve LaMAR/Kapture image path to an existing file.
    """
    shared_images_root = session_src / "raw_data"
    p = Path(image_path)

    # Fast path: as-is under scene/images.
    c0 = shared_images_root / p
    if c0.exists():
        return c0

    parts = list(p.parts)
    if len(parts) >= 2:
        first_variants = _normalize_session_token(parts[0])
        rest = parts[1:]

        rest_variants = [rest]
        if rest and rest[0] == "images":
            rest_variants.append(rest[1:])

        for first in first_variants:
            for r in rest_variants:
                c = shared_images_root / first
                if r:
                    c = c.joinpath(*r)
                if c.exists():
                    return c

    # Last fallback: relative to scene root exactly as listed.
    c_last = session_src / p
    if c_last.exists():
        return c_last
    return None


def _copy_other_sensors_data(session_src: Path, session_dst: Path) -> list[str]:
    """Copy non-image sensor metadata files for portability.

    We intentionally copy lightweight text metadata and avoid large raw blobs.
    """
    out_root = session_dst / "other_sensors_data"
    out_root.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    skip_names = {"images.txt", "trajectories.txt", "queries.txt"}

    for txt in sorted(session_src.glob("*.txt")):
        if txt.name in skip_names:
            continue
        dst = out_root / txt.name
        shutil.copy2(txt, dst)
        copied.append(f"other_sensors_data/{txt.name}")

    proc_src = session_src / "proc"
    if proc_src.exists() and proc_src.is_dir():
        proc_dst = out_root / "proc"
        proc_dst.mkdir(parents=True, exist_ok=True)
        for txt in sorted(proc_src.rglob("*.txt")):
            rel = txt.relative_to(proc_src)
            dst = proc_dst / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(txt, dst)
            copied.append(f"other_sensors_data/proc/{rel.as_posix()}")

    return copied


def _write_lines(path: Path, rows: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row}\n")


def _write_poses(path: Path, rows: list[tuple[str, Pose]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for image_name, pose in rows:
            f.write(
                f"{image_name} {pose.tx:.9f} {pose.ty:.9f} {pose.tz:.9f} "
                f"{pose.qx:.9f} {pose.qy:.9f} {pose.qz:.9f} {pose.qw:.9f}\n"
            )


def _build_image_timestamp_map(images: dict[Key, str]) -> dict[str, str]:
    """Map each image path to its raw timestamp token from LaMAR images.txt.

    If an image path appears multiple times (multi-sensor edge cases),
    keep the first occurrence to preserve deterministic behavior.
    """
    out: dict[str, str] = {}
    for (timestamp, _sensor_id), image_path in images.items():
        out.setdefault(image_path, timestamp)
    return out


def _build_timestamp_rows(image_names: list[str], image_to_timestamp: dict[str, str]) -> tuple[list[str], int]:
    rows: list[str] = []
    missing = 0
    for image_name in image_names:
        ts = image_to_timestamp.get(image_name)
        if ts is None:
            missing += 1
            continue
        rows.append(f"{image_name} {ts}")
    return rows, missing


def convert_scene(
    lamar_scene_root: Path,
    output_scene_root: Path,
    map_session: str,
    query_sessions: list[str],
    sensor_priority: list[str],
    use_alignment_poses_for_query: bool,
    copy_images: bool,
) -> dict[str, object]:
    sessions_root = lamar_scene_root / "sessions"

    if not sessions_root.exists():
        raise FileNotFoundError(f"Missing sessions root: {sessions_root}")
    if not (lamar_scene_root / "images").exists():
        raise FileNotFoundError(f"Missing shared images root: {lamar_scene_root / 'images'}")

    missing_source_counter: Counter[str] = Counter()

    summary: dict[str, object] = {
        "lamar_scene_root": str(lamar_scene_root),
        "output_scene_root": str(output_scene_root),
        "map_session": map_session,
        "query_sessions": query_sessions,
        "query_sets": {},
    }

    # Convert map
    map_src = sessions_root / map_session
    map_dst = output_scene_root / map_session
    map_images = _read_images_txt(map_src / "images.txt")
    map_traj = _read_trajectories(map_src / "trajectories.txt")
    map_image_to_timestamp = _build_image_timestamp_map(map_images)

    map_image_paths = list(map_images.values())
    map_materialized: list[str] = []
    map_linked = 0
    for image_path in map_image_paths:
        src = _resolve_image_source(map_src, image_path)
        if src is None:
            missing_source_counter["map"] += 1
            continue
        _safe_link_or_copy(src, map_dst / "images" / image_path, copy_images)
        map_linked += 1
        map_materialized.append(image_path)

    map_materialized_set = set(map_materialized)

    map_pose_rows: list[tuple[str, Pose]] = []
    for key, image_path in map_images.items():
        if image_path not in map_materialized_set:
            continue
        if key in map_traj:
            map_pose_rows.append((image_path, map_traj[key]))
        else:
            ts = key[0]
            candidates = [pose for (t, _d), pose in map_traj.items() if t == ts]
            if candidates:
                map_pose_rows.append((image_path, candidates[0]))

    _write_lines(map_dst / "images.txt", map_materialized)
    _write_poses(map_dst / "poses.txt", map_pose_rows)
    map_timestamp_rows, missing_map_timestamps = _build_timestamp_rows(
        map_materialized,
        map_image_to_timestamp,
    )
    _write_lines(map_dst / "timestamps.txt", map_timestamp_rows)

    summary["map"] = {
        "num_images": len(map_image_paths),
        "num_images_kept": len(map_materialized),
        "num_poses": len(map_pose_rows),
        "num_images_materialized": map_linked,
        "num_timestamps": len(map_timestamp_rows),
        "missing_timestamps": missing_map_timestamps,
    }

    # Convert query sessions
    query_sets_out: dict[str, object] = {}
    for query_session in query_sessions:
        query_src = sessions_root / query_session
        query_set_name = query_session
        query_dst = output_scene_root / query_set_name

        images = _read_images_txt(query_src / "images.txt")
        image_to_timestamp = _build_image_timestamp_map(images)
        traj_default = _read_trajectories(query_src / "trajectories.txt")
        traj_aligned = _read_trajectories(query_src / "proc" / "alignment_trajectories.txt")
        query_traj = traj_aligned if (use_alignment_poses_for_query and traj_aligned) else traj_default

        queries = _read_queries(query_src / "queries.txt")
        if not queries:
            queries = list(images.keys())

        resolved_query_images: list[str] = []
        missing_query_images = 0

        for qk in queries:
            image_path, image_key = _resolve_query_image(qk, images, sensor_priority)
            if image_path is None:
                missing_query_images += 1
                continue

            resolved_query_images.append(image_path)
            # pose = query_traj.get(qk)
            # if pose is None and image_key is not None:
            #     pose = query_traj.get(image_key)
            # if pose is None:
            #     ts = qk[0]
            #     candidates = [p for (t, _d), p in query_traj.items() if t == ts]
            #     if candidates:
            #         pose = candidates[0]

            # if pose is not None:
            #     resolved_pose_rows.append((image_path, pose))
            # else:
            #     missing_query_poses += 1
            
        resolved_pose_rows: list[tuple[str, Pose]] = []
        missing_query_poses = 0
        for key, image_path in images.items():
            if key in query_traj:
                resolved_pose_rows.append((image_path, query_traj[key]))
            else:
                ts = key[0]
                candidates = [pose for (t, _d), pose in query_traj.items() if t == ts]
                if candidates:
                    resolved_pose_rows.append((image_path, candidates[0]))
                else:
                    missing_query_poses += 1
        
        dedup_query_images = list(dict.fromkeys(resolved_query_images))
        query_linked = 0
        materialized_images: list[str] = []
        for image_path in images.values():
            src = _resolve_image_source(query_src, image_path)
            if src is None:
                missing_source_counter[query_set_name] += 1
                continue
            _safe_link_or_copy(src, query_dst / "images" / image_path, copy_images)
            query_linked += 1
            materialized_images.append(image_path)

        materialized_set = set(materialized_images)
        filtered_query_names = [p for p in dedup_query_images if p in materialized_set]
        filtered_pose_rows = [(p, pose) for (p, pose) in resolved_pose_rows if p in materialized_set]

        _write_lines(query_dst / "images.txt", materialized_images)
        _write_lines(query_dst / "query_names.txt", filtered_query_names)
        _write_poses(query_dst / "poses.txt", filtered_pose_rows)
        timestamp_rows, missing_timestamps = _build_timestamp_rows(
            materialized_images,
            image_to_timestamp,
        )
        _write_lines(query_dst / "timestamps.txt", timestamp_rows)

        copied_sensor_files = _copy_other_sensors_data(query_src, query_dst)

        query_sets_out[query_set_name] = {
            "num_images_total": len(images),
            "num_images_kept": len(materialized_images),
            "num_queries_selected": len(dedup_query_images),
            "num_queries_kept": len(filtered_query_names),
            "num_query_poses": len(filtered_pose_rows),
            "num_timestamps": len(timestamp_rows),
            "missing_query_images": missing_query_images,
            "missing_query_poses": missing_query_poses,
            "missing_timestamps": missing_timestamps,
            "num_images_materialized": query_linked,
            "pose_source": "alignment_trajectories"
            if (use_alignment_poses_for_query and traj_aligned)
            else "trajectories",
            "copied_sensor_files": copied_sensor_files,
        }

    summary["query_sets"] = query_sets_out
    summary["missing_source_images"] = dict(missing_source_counter)
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="convert-lamar-kapture",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lamar-scene-root", type=Path, required=True)
    parser.add_argument("--output-scene-root", type=Path, required=True)
    parser.add_argument("--map-session", type=str, default="map")
    parser.add_argument(
        "--query-sessions",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--sensor-priority",
        nargs="+",
        default=["cam_phone", "hetlf", "hetrf", "hetrr", "hetll"],
        help="Preferred sensor substrings when a query key is rig-level and does not directly map to an image key.",
    )
    parser.add_argument("--use-alignment-poses-for-query", action="store_true")
    parser.add_argument("--copy-images", action="store_true")

    args = parser.parse_args()

    summary = convert_scene(
        lamar_scene_root=args.lamar_scene_root,
        output_scene_root=args.output_scene_root,
        map_session=args.map_session,
        query_sessions=list(args.query_sessions),
        sensor_priority=list(args.sensor_priority),
        use_alignment_poses_for_query=bool(args.use_alignment_poses_for_query),
        copy_images=bool(args.copy_images),
    )

    summary_path = args.output_scene_root / "conversion_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"conversion summary: {summary_path}")


if __name__ == "__main__":
    main()