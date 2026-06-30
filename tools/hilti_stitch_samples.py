#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def _workspace_root() -> Path:
    # .../src/code/retrieval-evaluation/tools -> workspace root
    return Path(__file__).resolve().parents[4]


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
        raise RuntimeError(f"Invalid YAML at {path}")
    return data


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
    x = dirs[:, 0].astype(np.float64)
    y = dirs[:, 1].astype(np.float64)
    z = dirs[:, 2].astype(np.float64)
    rho_xy = np.hypot(x, y)
    theta = np.arctan2(rho_xy, z)
    f = 0.5 * (fx + fy)
    rho_px = f * _rho_from_theta(theta, k)

    nz = rho_xy > 1e-12
    ux = np.zeros_like(rho_xy)
    uy = np.zeros_like(rho_xy)
    ux[nz] = x[nz] / rho_xy[nz]
    uy[nz] = y[nz] / rho_xy[nz]
    u = cx + ux * rho_px
    v = cy + uy * rho_px
    return u, v, theta


def _build_equirect_dirs(out_w: int, out_h: int) -> np.ndarray:
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) / out_w
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) / out_h
    lon = xs[None, :] * 2.0 * np.pi - np.pi
    lat = ys[:, None] * np.pi - (np.pi / 2.0)
    lon_grid = np.repeat(lon, out_h, axis=0)
    lat_grid = np.repeat(lat, out_w, axis=1)
    cos_lat = np.cos(lat_grid)
    x = cos_lat * np.sin(lon_grid)
    y = np.sin(lat_grid)
    z = cos_lat * np.cos(lon_grid)
    return np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float64)


def _precompute_maps(cam0: dict, cam1: dict, out_w: int, out_h: int, sphere_radius_m: float):
    fx0, fy0, cx0, cy0 = [float(x) for x in cam0["intrinsics"]]
    fx1, fy1, cx1, cy1 = [float(x) for x in cam1["intrinsics"]]
    w0, h0 = [int(x) for x in cam0["resolution"]]
    w1, h1 = [int(x) for x in cam1["resolution"]]
    k0 = np.array(cam0.get("distortion_coeffs", []), dtype=np.float64)
    k1 = np.array(cam1.get("distortion_coeffs", []), dtype=np.float64)

    t0 = np.array(cam0["T_cam_imu"], dtype=np.float64)
    t1 = np.array(cam1["T_cam_imu"], dtype=np.float64)
    t_rel = t1 @ np.linalg.inv(t0)
    r01 = t_rel[:3, :3]
    tr01 = t_rel[:3, 3]

    dirs = _build_equirect_dirs(out_w, out_h)
    pts_world = dirs * sphere_radius_m
    pts_cam0 = pts_world
    pts_cam1 = (pts_world - tr01.reshape(1, 3)) @ r01.T

    u0, v0, theta0 = _dirs_to_pixels_kb(pts_cam0, fx0, fy0, cx0, cy0, k0)
    u1, v1, theta1 = _dirs_to_pixels_kb(pts_cam1, fx1, fy1, cx1, cy1, k1)

    map0_x = u0.reshape(out_h, out_w).astype(np.float32)
    map0_y = v0.reshape(out_h, out_w).astype(np.float32)
    map1_x = u1.reshape(out_h, out_w).astype(np.float32)
    map1_y = v1.reshape(out_h, out_w).astype(np.float32)

    # Keep only rays within modeled fisheye FOV to avoid unstable border projections.
    max_theta0 = np.deg2rad(float(cam0.get("fov_deg", 195.0)) * 0.5)
    max_theta1 = np.deg2rad(float(cam1.get("fov_deg", 195.0)) * 0.5)
    theta0_ok = theta0.reshape(out_h, out_w) <= (max_theta0 + 1e-9)
    theta1_ok = theta1.reshape(out_h, out_w) <= (max_theta1 + 1e-9)

    valid0 = (map0_x >= 0) & (map0_x < (w0 - 1)) & (map0_y >= 0) & (map0_y < (h0 - 1)) & theta0_ok
    valid1 = (map1_x >= 0) & (map1_x < (w1 - 1)) & (map1_y >= 0) & (map1_y < (h1 - 1)) & theta1_ok

    return {
        "map0_x": map0_x,
        "map0_y": map0_y,
        "map1_x": map1_x,
        "map1_y": map1_y,
        "valid0": valid0,
        "valid1": valid1,
    }


def _apply_static_masks(valid0: np.ndarray, valid1: np.ndarray, maps: dict, mask0: Path | None, mask1: Path | None):
    out0 = valid0.copy()
    out1 = valid1.copy()

    if mask0 is not None and mask0.exists():
        m0 = cv2.imread(str(mask0), cv2.IMREAD_GRAYSCALE)
        if m0 is not None:
            rem0 = cv2.remap(
                m0,
                maps["map0_x"],
                maps["map0_y"],
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
            out0 = out0 & (rem0 < 250)

    if mask1 is not None and mask1.exists():
        m1 = cv2.imread(str(mask1), cv2.IMREAD_GRAYSCALE)
        if m1 is not None:
            rem1 = cv2.remap(
                m1,
                maps["map1_x"],
                maps["map1_y"],
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
            out1 = out1 & (rem1 < 250)

    return out0, out1


def _build_feather_weights(valid0: np.ndarray, valid1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Feather by distance to invalid regions so overlaps blend more smoothly.
    m0 = (valid0.astype(np.uint8) * 255)
    m1 = (valid1.astype(np.uint8) * 255)
    d0 = cv2.distanceTransform(m0, cv2.DIST_L2, 3).astype(np.float32)
    d1 = cv2.distanceTransform(m1, cv2.DIST_L2, 3).astype(np.float32)
    denom = d0 + d1
    denom_safe = np.where(denom > 1e-6, denom, 1.0)
    w0 = np.where(denom > 1e-6, d0 / denom_safe, 0.0)
    w1 = np.where(denom > 1e-6, d1 / denom_safe, 0.0)
    return w0, w1


def _timestamp(name: str) -> float:
    return float(Path(name).stem)


def main() -> None:
    root = _workspace_root()
    default_hilti = root / "src" / "capture" / "Hilti"
    default_calib = root / "src" / "resources" / "hilti-trimble-slam-challenge-2026" / "config" / "hilti_openvins" / "kalibr_imucam_chain.yaml"
    default_mask0 = root / "src" / "resources" / "hilti-trimble-slam-challenge-2026" / "config" / "hilti_openvins" / "mask_cam0.png"
    default_mask1 = root / "src" / "resources" / "hilti-trimble-slam-challenge-2026" / "config" / "hilti_openvins" / "mask_cam1.png"
    default_out = root / "src" / "outputs" / "hilti_equirect" / "samples_stitch"

    p = argparse.ArgumentParser(description="Stitch a few Hilti cam0/cam1 frames into equirectangular images.")
    p.add_argument("--run-name", required=True, help="Run name, e.g. floor_2_2025-10-28_run_1")
    p.add_argument("--hilti-root", default=str(default_hilti))
    p.add_argument("--calib-yaml", default=str(default_calib))
    p.add_argument("--mask0", default=str(default_mask0))
    p.add_argument("--mask1", default=str(default_mask1))
    p.add_argument("--out-dir", default=str(default_out))
    p.add_argument("--num-samples", type=int, default=6)
    p.add_argument("--min-timestamp", type=float, default=10005.0)
    p.add_argument("--out-width", type=int, default=2048)
    p.add_argument("--out-height", type=int, default=1024)
    p.add_argument("--sphere-radius-m", type=float, default=10.0)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--rotate-cam0-180", action="store_true", help="Rotate cam0 image by 180 deg before stitching.")
    p.add_argument("--rotate-cam1-180", action="store_true", help="Rotate cam1 image by 180 deg before stitching.")
    p.add_argument("--blend-mode", choices=["average", "feather", "winner"], default="feather")
    p.add_argument("--roll-px", type=int, default=0, help="Horizontal pixel roll on final equirect to move seam.")
    args = p.parse_args()

    run_dir = Path(args.hilti_root) / "extracted_data" / args.run_name / "images"
    cam0_dir = run_dir / "cam0"
    cam1_dir = run_dir / "cam1"
    if not cam0_dir.exists() or not cam1_dir.exists():
        raise FileNotFoundError(f"Missing cam folders in {run_dir}")

    cfg = _read_yaml_flexible(Path(args.calib_yaml))
    cam0 = cfg.get("cam0")
    cam1 = cfg.get("cam1")
    if not isinstance(cam0, dict) or not isinstance(cam1, dict):
        raise RuntimeError("Calibration YAML must contain cam0 and cam1 entries")

    maps = _precompute_maps(cam0, cam1, args.out_width, args.out_height, args.sphere_radius_m)
    valid0, valid1 = _apply_static_masks(
        maps["valid0"],
        maps["valid1"],
        maps,
        Path(args.mask0) if args.mask0 else None,
        Path(args.mask1) if args.mask1 else None,
    )

    feather_w0 = None
    feather_w1 = None
    if args.blend_mode == "feather":
        feather_w0, feather_w1 = _build_feather_weights(valid0, valid1)

    names0 = {x.name for x in cam0_dir.glob("*.jpg")}
    names1 = {x.name for x in cam1_dir.glob("*.jpg")}
    names = [n for n in names0.intersection(names1) if _timestamp(n) >= args.min_timestamp]
    names.sort(key=_timestamp)
    if args.num_samples > 0:
        names = names[: args.num_samples]
    if not names:
        raise RuntimeError("No paired frames selected. Check run name and min timestamp.")

    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    quality = max(1, min(100, int(args.jpeg_quality)))

    written = []
    for idx, name in enumerate(names, 1):
        img0 = cv2.imread(str(cam0_dir / name), cv2.IMREAD_COLOR)
        img1 = cv2.imread(str(cam1_dir / name), cv2.IMREAD_COLOR)
        if img0 is None or img1 is None:
            continue

        if args.rotate_cam0_180:
            img0 = cv2.rotate(img0, cv2.ROTATE_180)
        if args.rotate_cam1_180:
            img1 = cv2.rotate(img1, cv2.ROTATE_180)

        w0 = cv2.remap(
            img0,
            maps["map0_x"],
            maps["map0_y"],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        w1 = cv2.remap(
            img1,
            maps["map1_x"],
            maps["map1_y"],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        if args.blend_mode == "feather" and feather_w0 is not None and feather_w1 is not None:
            a0 = feather_w0
            a1 = feather_w1
        else:
            a0 = valid0.astype(np.float32)
            a1 = valid1.astype(np.float32)

        if args.blend_mode == "winner":
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

        if args.roll_px != 0:
            pano = np.roll(pano, shift=int(args.roll_px), axis=1)

        out_name = f"{Path(name).stem}_eq.jpg"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), pano, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        written.append(out_name)
        print(f"[{idx}/{len(names)}] wrote {out_path}")

    meta = {
        "run_name": args.run_name,
        "num_selected_pairs": len(names),
        "num_written": len(written),
        "out_dir": str(out_dir),
        "params": {
            "min_timestamp": args.min_timestamp,
            "out_width": args.out_width,
            "out_height": args.out_height,
            "sphere_radius_m": args.sphere_radius_m,
            "jpeg_quality": quality,
            "rotate_cam0_180": bool(args.rotate_cam0_180),
            "rotate_cam1_180": bool(args.rotate_cam1_180),
            "blend_mode": args.blend_mode,
            "roll_px": int(args.roll_px),
        },
        "files": written,
    }
    (out_dir / "stitch_samples_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Done. Generated {len(written)} images in {out_dir}")


if __name__ == "__main__":
    main()
