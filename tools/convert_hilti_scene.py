from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class Pose:
    tx: float
    ty: float
    tz: float
    qx: float
    qy: float
    qz: float
    qw: float


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _scene_from_run(run_name: str) -> str:
    # floor_2_2025-10-28_run_1 -> floor_2
    if not run_name.startswith("floor_"):
        raise ValueError(f"Unexpected run name: {run_name}")
    parts = run_name.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected run name: {run_name}")
    return f"{parts[0]}_{parts[1]}"


def _safe_link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_images:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def _read_cam0_poses_csv(path: Path) -> tuple[dict[str, Pose], dict[str, Pose]]:
    exact: dict[str, Pose] = {}
    rounded6: dict[str, Pose] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row["timestamp"].strip()
            pose = Pose(
                tx=float(row["tx"]),
                ty=float(row["ty"]),
                tz=float(row["tz"]),
                qx=float(row["qx"]),
                qy=float(row["qy"]),
                qz=float(row["qz"]),
                qw=float(row["qw"]),
            )
            exact[ts] = pose
            rounded6[f"{float(ts):.6f}"] = pose
    return exact, rounded6


def _resolve_pose(stem_ts: str, poses_exact: dict[str, Pose], poses_rounded6: dict[str, Pose]) -> Pose | None:
    if stem_ts in poses_exact:
        return poses_exact[stem_ts]
    return poses_rounded6.get(f"{float(stem_ts):.6f}")


def _pose_to_rt(p: Pose) -> tuple[np.ndarray, np.ndarray]:
    x, y, z, w = p.qx, p.qy, p.qz, p.qw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    t = np.array([p.tx, p.ty, p.tz], dtype=np.float64)
    return r, t


def _rot_to_quat_xyzw(r: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    return qx, qy, qz, qw


def _apply_cam0_to_cam1(p: Pose, t_cam0_cam1: np.ndarray, r_cam0_cam1: np.ndarray) -> Pose:
    r_map_cam0, t_map_cam0 = _pose_to_rt(p)
    r_map_cam1 = r_map_cam0 @ r_cam0_cam1
    t_map_cam1 = t_map_cam0 + r_map_cam0 @ t_cam0_cam1
    qx, qy, qz, qw = _rot_to_quat_xyzw(r_map_cam1)
    return Pose(
        tx=float(t_map_cam1[0]),
        ty=float(t_map_cam1[1]),
        tz=float(t_map_cam1[2]),
        qx=float(qx),
        qy=float(qy),
        qz=float(qz),
        qw=float(qw),
    )


def _read_yaml_flexible(path: Path) -> dict:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin1")
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("%YAML"):
        text = "\n".join(lines[1:])
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid YAML: {path}")
    return data


def _read_cam0_cam1_extrinsic(calib_yaml: Path) -> tuple[np.ndarray, np.ndarray]:
    data = _read_yaml_flexible(calib_yaml)
    cam0 = data["cam0"]
    cam1 = data["cam1"]
    t0 = np.array(cam0["T_cam_imu"], dtype=np.float64)
    t1 = np.array(cam1["T_cam_imu"], dtype=np.float64)
    t_rel = t1 @ np.linalg.inv(t0)
    return t_rel[:3, 3], t_rel[:3, :3]


def _timestamp_key(name: str) -> float:
    return float(Path(name).stem)


def _write_lines(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r}\n")


def _write_json(path: Path, data: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_poses(path: Path, rows: list[tuple[str, Pose]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for name, p in rows:
            f.write(
                f"{name} {p.tx:.9f} {p.ty:.9f} {p.tz:.9f} {p.qx:.9f} {p.qy:.9f} {p.qz:.9f} {p.qw:.9f}\n"
            )


def _rho_from_theta(theta: np.ndarray, k: np.ndarray) -> np.ndarray:
    rho = theta.copy()
    if k.size == 0:
        return rho
    t2 = theta * theta
    t_pow = theta * t2
    for ki in k.flat:
        rho = rho + ki * t_pow
        t_pow = t_pow * t2
    return rho


def _dirs_to_pixels_kb(dirs: np.ndarray, fx: float, fy: float, cx: float, cy: float, k: np.ndarray):
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    rho_xy = np.hypot(x, y)
    theta = np.arctan2(rho_xy, z)
    f = 0.5 * (fx + fy)
    rho_px = f * _rho_from_theta(theta, k)
    nz = rho_xy > 1e-12
    ux = np.zeros_like(rho_xy)
    uy = np.zeros_like(rho_xy)
    ux[nz] = x[nz] / rho_xy[nz]
    uy[nz] = y[nz] / rho_xy[nz]
    return cx + ux * rho_px, cy + uy * rho_px


def _build_equirect_dirs(out_w: int, out_h: int) -> np.ndarray:
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) / out_w
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) / out_h
    lon = xs[None, :] * (2.0 * np.pi) - np.pi
    lat = ys[:, None] * np.pi - (np.pi / 2.0)
    lon_grid = np.repeat(lon, out_h, axis=0)
    lat_grid = np.repeat(lat, out_w, axis=1)
    cos_lat = np.cos(lat_grid)
    x = cos_lat * np.sin(lon_grid)
    y = np.sin(lat_grid)
    z = cos_lat * np.cos(lon_grid)
    return np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float64)


def _prepare_stitch_maps(calib_yaml: Path, out_w: int, out_h: int, sphere_radius_m: float, mask0: Path, mask1: Path):
    cfg = _read_yaml_flexible(calib_yaml)
    cam0 = cfg["cam0"]
    cam1 = cfg["cam1"]
    fx0, fy0, cx0, cy0 = [float(x) for x in cam0["intrinsics"]]
    fx1, fy1, cx1, cy1 = [float(x) for x in cam1["intrinsics"]]
    w0, h0 = [int(x) for x in cam0["resolution"]]
    w1, h1 = [int(x) for x in cam1["resolution"]]
    k0 = np.array(cam0.get("distortion_coeffs", []), dtype=np.float64)
    k1 = np.array(cam1.get("distortion_coeffs", []), dtype=np.float64)
    t01, r01 = _read_cam0_cam1_extrinsic(calib_yaml)

    dirs = _build_equirect_dirs(out_w, out_h)
    pts_world = dirs * sphere_radius_m
    pts_cam0 = pts_world
    pts_cam1 = (pts_world - t01.reshape(1, 3)) @ r01.T
    u0, v0 = _dirs_to_pixels_kb(pts_cam0, fx0, fy0, cx0, cy0, k0)
    u1, v1 = _dirs_to_pixels_kb(pts_cam1, fx1, fy1, cx1, cy1, k1)

    map0_x = u0.reshape(out_h, out_w).astype(np.float32)
    map0_y = v0.reshape(out_h, out_w).astype(np.float32)
    map1_x = u1.reshape(out_h, out_w).astype(np.float32)
    map1_y = v1.reshape(out_h, out_w).astype(np.float32)

    valid0 = (map0_x >= 0) & (map0_x < (w0 - 1)) & (map0_y >= 0) & (map0_y < (h0 - 1))
    valid1 = (map1_x >= 0) & (map1_x < (w1 - 1)) & (map1_y >= 0) & (map1_y < (h1 - 1))

    if mask0.exists():
        m0 = cv2.imread(str(mask0), cv2.IMREAD_GRAYSCALE)
        if m0 is not None:
            rem0 = cv2.remap(m0, map0_x, map0_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            valid0 = valid0 & (rem0 < 250)

    if mask1.exists():
        m1 = cv2.imread(str(mask1), cv2.IMREAD_GRAYSCALE)
        if m1 is not None:
            rem1 = cv2.remap(m1, map1_x, map1_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            valid1 = valid1 & (rem1 < 250)

    return {
        "map0_x": map0_x,
        "map0_y": map0_y,
        "map1_x": map1_x,
        "map1_y": map1_y,
        "valid0": valid0.astype(np.float32),
        "valid1": valid1.astype(np.float32),
    }


def _build_feather_weights(valid0: np.ndarray, valid1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m0 = (valid0 > 0).astype(np.uint8) * 255
    m1 = (valid1 > 0).astype(np.uint8) * 255
    d0 = cv2.distanceTransform(m0, cv2.DIST_L2, 3).astype(np.float32)
    d1 = cv2.distanceTransform(m1, cv2.DIST_L2, 3).astype(np.float32)
    denom = d0 + d1
    denom_safe = np.where(denom > 1e-6, denom, 1.0)
    w0 = np.where(denom > 1e-6, d0 / denom_safe, 0.0)
    w1 = np.where(denom > 1e-6, d1 / denom_safe, 0.0)
    return w0, w1


def _stitch_pair(
    cam0_path: Path,
    cam1_path: Path,
    maps: dict,
    jpeg_quality: int,
    *,
    rotate_cam0_180: bool,
    rotate_cam1_180: bool,
    blend_mode: str,
    roll_px: int,
) -> np.ndarray:
    img0 = cv2.imread(str(cam0_path), cv2.IMREAD_COLOR)
    img1 = cv2.imread(str(cam1_path), cv2.IMREAD_COLOR)
    if img0 is None or img1 is None:
        raise RuntimeError(f"Failed to read image pair: {cam0_path}, {cam1_path}")

    if rotate_cam0_180:
        img0 = cv2.rotate(img0, cv2.ROTATE_180)
    if rotate_cam1_180:
        img1 = cv2.rotate(img1, cv2.ROTATE_180)

    w0 = cv2.remap(img0, maps["map0_x"], maps["map0_y"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    w1 = cv2.remap(img1, maps["map1_x"], maps["map1_y"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if blend_mode == "feather" and "feather_w0" in maps and "feather_w1" in maps:
        a0 = maps["feather_w0"]
        a1 = maps["feather_w1"]
    else:
        a0 = maps["valid0"]
        a1 = maps["valid1"]

    if blend_mode == "winner":
        pick0 = (a0 >= a1) & ((a0 > 0) | (a1 > 0))
        pick1 = (~pick0) & ((a0 > 0) | (a1 > 0))
        pano = np.zeros_like(w0, dtype=np.uint8)
        pano[pick0] = w0[pick0]
        pano[pick1] = w1[pick1]
    else:
        denom = a0 + a1
        denom_safe = np.where(denom > 1e-6, denom, 1.0)
        pano = (w0.astype(np.float32) * a0[..., None] + w1.astype(np.float32) * a1[..., None]) / denom_safe[..., None]
        pano[denom <= 1e-6] = 0
        pano = np.clip(pano, 0, 255).astype(np.uint8)

    if roll_px != 0:
        pano = np.roll(pano, shift=int(roll_px), axis=1)

    if jpeg_quality > 0:
        # Keep consistent encoding behavior with saved JPG outputs.
        ok, enc = cv2.imencode(".jpg", pano, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if ok:
            pano = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return pano


def _collect_scene_runs(extracted_data_root: Path, scene: str) -> list[str]:
    runs: list[str] = []
    for p in sorted(extracted_data_root.iterdir()):
        if not p.is_dir():
            continue
        if _scene_from_run(p.name) == scene:
            runs.append(p.name)
    return runs


def _sample_queries_for_equirect(
    frames_by_run: dict[str, list[str]],
    *,
    seq_len: int,
    stride: int,
    min_sequences: int,
    low_resource_threshold: int,
) -> tuple[list[str], list[list[str]], dict[str, Any]]:
    seq_len = max(1, seq_len)
    stride = max(1, stride)

    total_frames = sum(len(v) for v in frames_by_run.values())
    fallback_mode = "normal"
    if total_frames < low_resource_threshold:
        fallback_mode = "low-resource-b"
        seq_len = 1
        stride = 1

    sampled_by_run: dict[str, list[str]] = {}
    for run_name, names in frames_by_run.items():
        if not names:
            sampled_by_run[run_name] = []
            continue
        sampled = names[::stride]
        if not sampled:
            sampled = [names[0]]
        sampled_by_run[run_name] = sampled

    queries = sorted({x for rows in sampled_by_run.values() for x in rows})
    sequences: list[list[str]] = []
    seq_window_stride = max(1, seq_len // 2)
    for run_name in sorted(sampled_by_run):
        rows = sampled_by_run[run_name]
        if not rows:
            continue
        if len(rows) <= seq_len:
            sequences.append(rows)
            continue
        for i in range(0, len(rows) - seq_len + 1, seq_window_stride):
            sequences.append(rows[i : i + seq_len])

    if len(sequences) < min_sequences and fallback_mode == "normal":
        fallback_mode = "low-resource-a"

    sampling_manifest = {
        "mode": fallback_mode,
        "total_frames": total_frames,
        "query_count": len(queries),
        "sequence_count": len(sequences),
        "seq_len": seq_len,
        "stride": stride,
        "min_sequences_target": min_sequences,
    }
    return queries, sequences, sampling_manifest


def _export_scene(
    *,
    dataset_type: str,
    scene: str,
    extracted_data_root: Path,
    dataset_root: Path,
    output_dataset_name: str,
    scene_runs: list[str],
    min_timestamp: float,
    single_pose_policy: str,
    t_cam0_cam1: np.ndarray,
    r_cam0_cam1: np.ndarray,
    copy_images: bool,
    stitch_maps: dict | None,
    jpeg_quality: int,
    max_frames: int,
    eq_seq_len: int,
    eq_stride: int,
    eq_min_sequences: int,
    low_resource_threshold: int,
    rotate_cam0_180: bool,
    rotate_cam1_180: bool,
    blend_mode: str,
    roll_px: int,
) -> dict:
    scene_root = dataset_root / output_dataset_name / scene
    map_images_root = scene_root / "map" / "images"
    query_set_name = "query_cam1" if dataset_type == "single" else "query"
    query_images_root = scene_root / query_set_name / "images"

    map_lines: list[str] = []
    query_lines: list[str] = []
    map_poses: list[tuple[str, Pose]] = []
    query_poses: list[tuple[str, Pose]] = []
    map_ts_lines: list[str] = []
    query_ts_lines: list[str] = []

    map_written = 0
    query_written = 0
    max_per_partition = max_frames if max_frames > 0 else 10**18
    eq_candidates_by_run: dict[str, list[str]] = {}

    for run_name in sorted(scene_runs):
        run_root = extracted_data_root / run_name
        cam0_dir = run_root / "images" / "cam0"
        cam1_dir = run_root / "images" / "cam1"
        poses_exact, poses_rounded6 = _read_cam0_poses_csv(run_root / "cam0_poses.csv")
        common_names = [
            p.name
            for p in cam0_dir.glob("*.jpg")
            if (cam1_dir / p.name).exists()
            and _resolve_pose(Path(p.name).stem, poses_exact, poses_rounded6) is not None
            and _timestamp_key(p.name) >= min_timestamp
        ]
        common_names.sort(key=_timestamp_key)

        for name in common_names:
            rel = f"{run_name}/{name}" if dataset_type == "single" else f"{run_name}/{Path(name).stem}_eq.jpg"
            dst = map_images_root / rel
            pose = _resolve_pose(Path(name).stem, poses_exact, poses_rounded6)
            if pose is None:
                continue
            ts = _timestamp_key(name)

            if dataset_type == "single":
                _safe_link_or_copy(cam0_dir / name, dst, copy_images)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                pano = _stitch_pair(
                    cam0_dir / name,
                    cam1_dir / name,
                    stitch_maps,
                    jpeg_quality,
                    rotate_cam0_180=rotate_cam0_180,
                    rotate_cam1_180=rotate_cam1_180,
                    blend_mode=blend_mode,
                    roll_px=roll_px,
                )
                cv2.imwrite(str(dst), pano, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])

            map_lines.append(rel)
            map_poses.append((rel, pose))
            map_ts_lines.append(f"{rel} {ts:.6f}")
            map_written += 1
            if map_written >= max_per_partition:
                break

            if dataset_type == "equirect":
                eq_candidates_by_run.setdefault(run_name, []).append(rel)

        if dataset_type == "single":
            for name in common_names:
                rel = f"{run_name}/{name}"
                dst = query_images_root / rel
                base_pose = _resolve_pose(Path(name).stem, poses_exact, poses_rounded6)
                if base_pose is None:
                    continue
                ts = _timestamp_key(name)
                _safe_link_or_copy(cam1_dir / name, dst, copy_images)
                pose = base_pose if single_pose_policy == "v1" else _apply_cam0_to_cam1(base_pose, t_cam0_cam1, r_cam0_cam1)

                query_lines.append(rel)
                query_poses.append((rel, pose))
                query_ts_lines.append(f"{rel} {ts:.6f}")
                query_written += 1
                if query_written >= max_per_partition:
                    break

    map_lines = sorted(dict.fromkeys(map_lines))
    sampling_manifest: dict[str, Any] | None = None

    if dataset_type == "equirect":
        sampled_queries, sequences, sampling_manifest = _sample_queries_for_equirect(
            eq_candidates_by_run,
            seq_len=eq_seq_len,
            stride=eq_stride,
            min_sequences=eq_min_sequences,
            low_resource_threshold=low_resource_threshold,
        )
        query_lines = sampled_queries
        map_pose_dict = {k: v for k, v in map_poses}
        map_ts_dict = {r.split(" ", 1)[0]: r for r in map_ts_lines}
        query_poses = [(k, map_pose_dict[k]) for k in query_lines if k in map_pose_dict]
        query_ts_lines = [map_ts_dict[k] for k in query_lines if k in map_ts_dict]
    else:
        query_lines = sorted(dict.fromkeys(query_lines))

    map_pose_dict = {k: v for k, v in map_poses}
    query_pose_dict = {k: v for k, v in query_poses}
    map_ts_dict = {r.split(" ", 1)[0]: r for r in map_ts_lines}
    query_ts_dict = {r.split(" ", 1)[0]: r for r in query_ts_lines}

    _write_lines(scene_root / "map" / "images.txt", map_lines)
    _write_poses(scene_root / "map" / "poses.txt", [(k, map_pose_dict[k]) for k in map_lines if k in map_pose_dict])
    _write_lines(scene_root / "map" / "timestamps.txt", [map_ts_dict[k] for k in map_lines if k in map_ts_dict])

    _write_lines(scene_root / query_set_name / "images.txt", query_lines)
    _write_lines(scene_root / query_set_name / "query_names.txt", query_lines)
    _write_poses(scene_root / query_set_name / "poses.txt", [(k, query_pose_dict[k]) for k in query_lines if k in query_pose_dict])
    _write_lines(scene_root / query_set_name / "timestamps.txt", [query_ts_dict[k] for k in query_lines if k in query_ts_dict])
    if dataset_type == "equirect" and sampling_manifest is not None:
        _write_json(scene_root / query_set_name / "query_sequences.json", sequences)

    manifest = {
        "dataset_type": dataset_type,
        "scene": scene,
        "scene_runs": sorted(scene_runs),
        "split_mode": "floor_global",
        "single_pose_policy": single_pose_policy if dataset_type == "single" else "cam0_proxy",
        "stitch": {
            "rotate_cam0_180": bool(rotate_cam0_180),
            "rotate_cam1_180": bool(rotate_cam1_180),
            "blend_mode": blend_mode,
            "roll_px": int(roll_px),
        }
        if dataset_type == "equirect"
        else None,
        "counts": {
            "map_images": len(map_lines),
            "query_images": len(query_lines),
        },
    }
    if manifest.get("stitch") is None:
        manifest.pop("stitch", None)
    meta_dir = scene_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    _write_json(meta_dir / "scene_manifest.json", manifest)
    if sampling_manifest is not None:
        _write_json(meta_dir / "sampling_manifest.json", sampling_manifest)
    return manifest


def main() -> None:
    root = _workspace_root()
    default_hilti_root = root / "src" / "capture" / "Hilti"
    default_dataset_root = root / "src" / "code" / "retrieval-evaluation" / "datasets"
    default_calib_yaml = root / "src" / "resources" / "hilti-trimble-slam-challenge-2026" / "config" / "hilti_openvins" / "kalibr_imucam_chain.yaml"
    default_mask0 = root / "src" / "resources" / "hilti-trimble-slam-challenge-2026" / "config" / "hilti_openvins" / "mask_cam0.png"
    default_mask1 = root / "src" / "resources" / "hilti-trimble-slam-challenge-2026" / "config" / "hilti_openvins" / "mask_cam1.png"

    p = argparse.ArgumentParser(description="Convert Hilti scene to retrieval-evaluation dataset format.")
    p.add_argument("--dataset-type", required=True, choices=["single", "equirect"])
    p.add_argument("--scene", required=True, choices=["floor_1", "floor_2", "floor_UG1"])
    p.add_argument("--hilti-root", default=str(default_hilti_root))
    p.add_argument("--dataset-root", default=str(default_dataset_root))
    p.add_argument("--output-dataset-name", default="")
    p.add_argument("--split-policy", default="floor-global", choices=["floor-global"], help="Protocol is fixed to floor-global for current Hilti conversion")
    p.add_argument("--single-pose-policy", default="v2", choices=["v1", "v2"])
    p.add_argument("--min-timestamp", type=float, default=10005.0)
    p.add_argument("--copy-images", action="store_true")
    p.add_argument("--run", action="append", default=[], help="Optional explicit runs to include. By default includes all runs in floor")
    p.add_argument("--max-frames", type=int, default=0)

    p.add_argument("--calib-yaml", default=str(default_calib_yaml))
    p.add_argument("--mask0", default=str(default_mask0))
    p.add_argument("--mask1", default=str(default_mask1))
    p.add_argument("--out-width", type=int, default=2048)
    p.add_argument("--out-height", type=int, default=1024)
    p.add_argument("--sphere-radius-m", type=float, default=10.0)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--rotate-cam0-180", action="store_true")
    p.add_argument("--rotate-cam1-180", action="store_true")
    p.add_argument("--blend-mode", choices=["average", "feather", "winner"], default="feather")
    p.add_argument("--roll-px", type=int, default=512)
    p.add_argument("--eq-seq-len", type=int, default=8)
    p.add_argument("--eq-stride", type=int, default=10)
    p.add_argument("--eq-min-sequences", type=int, default=30)
    p.add_argument("--low-resource-threshold", type=int, default=120)
    args = p.parse_args()

    hilti_root = Path(args.hilti_root)
    extracted_data_root = hilti_root / "extracted_data"
    dataset_root = Path(args.dataset_root)
    output_dataset_name = args.output_dataset_name or ("hilti_single" if args.dataset_type == "single" else "hilti_equirect")

    scene_runs = _collect_scene_runs(extracted_data_root, args.scene)
    if not scene_runs:
        raise RuntimeError(f"No runs found for scene {args.scene}")

    selected_runs = sorted(args.run) if args.run else scene_runs

    t_cam0_cam1, r_cam0_cam1 = _read_cam0_cam1_extrinsic(Path(args.calib_yaml))
    stitch_maps = None
    if args.dataset_type == "equirect":
        stitch_maps = _prepare_stitch_maps(
            calib_yaml=Path(args.calib_yaml),
            out_w=args.out_width,
            out_h=args.out_height,
            sphere_radius_m=args.sphere_radius_m,
            mask0=Path(args.mask0),
            mask1=Path(args.mask1),
        )
        if args.blend_mode == "feather":
            fw0, fw1 = _build_feather_weights(stitch_maps["valid0"], stitch_maps["valid1"])
            stitch_maps["feather_w0"] = fw0
            stitch_maps["feather_w1"] = fw1

    manifest = _export_scene(
        dataset_type=args.dataset_type,
        scene=args.scene,
        extracted_data_root=extracted_data_root,
        dataset_root=dataset_root,
        output_dataset_name=output_dataset_name,
        scene_runs=selected_runs,
        min_timestamp=args.min_timestamp,
        single_pose_policy=args.single_pose_policy,
        t_cam0_cam1=t_cam0_cam1,
        r_cam0_cam1=r_cam0_cam1,
        copy_images=args.copy_images,
        stitch_maps=stitch_maps,
        jpeg_quality=max(1, min(100, int(args.jpeg_quality))),
        max_frames=max(0, int(args.max_frames)),
        eq_seq_len=max(1, int(args.eq_seq_len)),
        eq_stride=max(1, int(args.eq_stride)),
        eq_min_sequences=max(1, int(args.eq_min_sequences)),
        low_resource_threshold=max(1, int(args.low_resource_threshold)),
        rotate_cam0_180=bool(args.rotate_cam0_180),
        rotate_cam1_180=bool(args.rotate_cam1_180),
        blend_mode=str(args.blend_mode),
        roll_px=int(args.roll_px),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
