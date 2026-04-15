#!/usr/bin/env bash
set -euo pipefail

# Convert Hilti for all scenes using floor-global protocol.
# Supports: single, equirect, or both.

DATASET_MODE="single"            # single | equirect | both
PY_ENV="vnu-ar"
SCENES="floor_1 floor_2 floor_UG1"
# Default: write to thesis/experiments/datasets/hilti
# (script cwd is experiments/code/retrieval-evaluation)
DATASET_ROOT="../../datasets/hilti"
HILTI_ROOT=""
CALIB_YAML=""
MASK0=""
MASK1=""
MAX_FRAMES=0
CLEAN_OUTPUT=0

# Map sampling defaults (per-run).
# Proposed defaults:
# - single map: dt=1.0 cap=1200
# - equirect map: dt=1.0 cap=600
SINGLE_MAP_SAMPLE_DT_S=1.0
SINGLE_MAP_MAX_PER_RUN=1200
EQUIRECT_MAP_SAMPLE_DT_S=1.0
EQUIRECT_MAP_MAX_PER_RUN=600

# Equirect stitching defaults (aligned with validated command).
ROTATE_CAM0_180=1
ROTATE_CAM1_180=1
BLEND_MODE="feather"           # average | feather | winner
ROLL_PX=512
OUT_WIDTH=2048
OUT_HEIGHT=1024
SPHERE_RADIUS_M=10.0
JPEG_QUALITY=95

# Equirect query sampling defaults.
EQ_SEQ_LEN=8
EQ_STRIDE=10
EQ_MIN_SEQUENCES=30
LOW_RESOURCE_THRESHOLD=120

# Single-camera query sampling defaults.
SINGLE_SEQ_MODE="time"          # time | frames
SINGLE_SAMPLE_DT_S=1.0          # time mode: sample one query every dt seconds
SINGLE_SEQ_DURATION_S=10.0      # time mode: chunk per-run sequences by duration in seconds

# Frame-based single sampling options (used only when SINGLE_SEQ_MODE=frames).
SINGLE_SEQ_LEN=0                # 0 = full per-run sequences
SINGLE_STRIDE=1
SINGLE_MIN_SEQUENCES=1
SINGLE_SEQ_WINDOW_STRIDE=0

usage() {
  cat <<'EOF'
Usage: ./scripts/run_hilti_convert_all_scenes.sh [options]

Options:
  --mode <single|equirect|both>   Dataset mode to convert (default: both)
  --dataset-root <path>            Output dataset root (default: ../../datasets/hilti)
  --hilti-root <path>              Override Hilti root
  --calib-yaml <path>              Override calibration YAML
  --mask0 <path>                   Override mask_cam0 path
  --mask1 <path>                   Override mask_cam1 path
  --max-frames <int>               Limit frames per partition for smoke runs (0 = full)
  --clean-output                   Delete existing <dataset>/<scene> outputs before writing new artifacts
  --scenes "a b c"                 Space-separated scene list (default: floor_1 floor_2 floor_UG1)
  --py-env <name>                  Conda env name (default: vnu-ar)

Equirect options:
  --no-rotate-cam0                 Disable cam0 180-degree rotation
  --no-rotate-cam1                 Disable cam1 180-degree rotation
  --blend-mode <average|feather|winner>
  --roll-px <int>
  --out-width <int>
  --out-height <int>
  --sphere-radius-m <float>
  --jpeg-quality <int>

Sequence sampling options:
  --eq-seq-len <int>
  --eq-stride <int>
  --eq-min-sequences <int>
  --single-seq-mode <time|frames>
  --single-sample-dt-s <float>
  --single-seq-duration-s <float>
  --single-seq-len <int>            0 = full per-run sequences (backward compatible)
  --single-stride <int>
  --single-min-sequences <int>
  --single-seq-window-stride <int>  0 = default (seq_len//2) when seq_len>0
  --low-resource-threshold <int>

Map sampling options:
  --single-map-sample-dt-s <float>   Per-run map sampling dt for single (default: 1.0; 0 = disabled)
  --single-map-max-per-run <int>     Per-run map cap for single (default: 1200; 0 = no cap)
  --equirect-map-sample-dt-s <float> Per-run map sampling dt for equirect (default: 1.0; 0 = disabled)
  --equirect-map-max-per-run <int>   Per-run map cap for equirect (default: 600; 0 = no cap)

Examples:
  ./scripts/run_hilti_convert_all_scenes.sh
  ./scripts/run_hilti_convert_all_scenes.sh --mode single --dataset-root ./datasets
  ./scripts/run_hilti_convert_all_scenes.sh --mode equirect --max-frames 200
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) DATASET_MODE="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --hilti-root) HILTI_ROOT="$2"; shift 2 ;;
    --calib-yaml) CALIB_YAML="$2"; shift 2 ;;
    --mask0) MASK0="$2"; shift 2 ;;
    --mask1) MASK1="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --clean-output) CLEAN_OUTPUT=1; shift ;;
    --scenes) SCENES="$2"; shift 2 ;;
    --py-env) PY_ENV="$2"; shift 2 ;;

    --no-rotate-cam0) ROTATE_CAM0_180=0; shift ;;
    --no-rotate-cam1) ROTATE_CAM1_180=0; shift ;;
    --blend-mode) BLEND_MODE="$2"; shift 2 ;;
    --roll-px) ROLL_PX="$2"; shift 2 ;;
    --out-width) OUT_WIDTH="$2"; shift 2 ;;
    --out-height) OUT_HEIGHT="$2"; shift 2 ;;
    --sphere-radius-m) SPHERE_RADIUS_M="$2"; shift 2 ;;
    --jpeg-quality) JPEG_QUALITY="$2"; shift 2 ;;

    --eq-seq-len) EQ_SEQ_LEN="$2"; shift 2 ;;
    --eq-stride) EQ_STRIDE="$2"; shift 2 ;;
    --eq-min-sequences) EQ_MIN_SEQUENCES="$2"; shift 2 ;;
    --single-seq-mode) SINGLE_SEQ_MODE="$2"; shift 2 ;;
    --single-sample-dt-s) SINGLE_SAMPLE_DT_S="$2"; shift 2 ;;
    --single-seq-duration-s) SINGLE_SEQ_DURATION_S="$2"; shift 2 ;;
    --single-seq-len) SINGLE_SEQ_LEN="$2"; shift 2 ;;
    --single-stride) SINGLE_STRIDE="$2"; shift 2 ;;
    --single-min-sequences) SINGLE_MIN_SEQUENCES="$2"; shift 2 ;;
    --single-seq-window-stride) SINGLE_SEQ_WINDOW_STRIDE="$2"; shift 2 ;;
    --low-resource-threshold) LOW_RESOURCE_THRESHOLD="$2"; shift 2 ;;

    --single-map-sample-dt-s) SINGLE_MAP_SAMPLE_DT_S="$2"; shift 2 ;;
    --single-map-max-per-run) SINGLE_MAP_MAX_PER_RUN="$2"; shift 2 ;;
    --equirect-map-sample-dt-s) EQUIRECT_MAP_SAMPLE_DT_S="$2"; shift 2 ;;
    --equirect-map-max-per-run) EQUIRECT_MAP_MAX_PER_RUN="$2"; shift 2 ;;

    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ "$DATASET_MODE" != "single" && "$DATASET_MODE" != "equirect" && "$DATASET_MODE" != "both" ]]; then
  echo "Invalid --mode: $DATASET_MODE"
  exit 1
fi

run_convert() {
  local dataset_type="$1"
  local scene="$2"

  local cmd=(
    /home/long/miniconda3/bin/conda run -n "$PY_ENV" python
    tools/convert_hilti_scene.py
    --dataset-type "$dataset_type"
    --scene "$scene"
    --dataset-root "$DATASET_ROOT"
    --max-frames "$MAX_FRAMES"
  )

  if [[ "$CLEAN_OUTPUT" == "1" ]]; then
    cmd+=(--clean-output)
  fi

  if [[ -n "$HILTI_ROOT" ]]; then
    cmd+=(--hilti-root "$HILTI_ROOT")
  fi
  if [[ -n "$CALIB_YAML" ]]; then
    cmd+=(--calib-yaml "$CALIB_YAML")
  fi
  if [[ -n "$MASK0" ]]; then
    cmd+=(--mask0 "$MASK0")
  fi
  if [[ -n "$MASK1" ]]; then
    cmd+=(--mask1 "$MASK1")
  fi

  if [[ "$dataset_type" == "equirect" ]]; then
    cmd+=(
      --map-sample-dt-s "$EQUIRECT_MAP_SAMPLE_DT_S"
      --map-max-per-run "$EQUIRECT_MAP_MAX_PER_RUN"
      --blend-mode "$BLEND_MODE"
      --roll-px "$ROLL_PX"
      --out-width "$OUT_WIDTH"
      --out-height "$OUT_HEIGHT"
      --sphere-radius-m "$SPHERE_RADIUS_M"
      --jpeg-quality "$JPEG_QUALITY"
      --eq-seq-len "$EQ_SEQ_LEN"
      --eq-stride "$EQ_STRIDE"
      --eq-min-sequences "$EQ_MIN_SEQUENCES"
      --low-resource-threshold "$LOW_RESOURCE_THRESHOLD"
    )

    if [[ "$ROTATE_CAM0_180" == "1" ]]; then
      cmd+=(--rotate-cam0-180)
    fi
    if [[ "$ROTATE_CAM1_180" == "1" ]]; then
      cmd+=(--rotate-cam1-180)
    fi
  fi

  if [[ "$dataset_type" == "single" ]]; then
    cmd+=(
      --map-sample-dt-s "$SINGLE_MAP_SAMPLE_DT_S"
      --map-max-per-run "$SINGLE_MAP_MAX_PER_RUN"
      --single-seq-mode "$SINGLE_SEQ_MODE"
      --single-sample-dt-s "$SINGLE_SAMPLE_DT_S"
      --single-seq-duration-s "$SINGLE_SEQ_DURATION_S"
      --single-seq-len "$SINGLE_SEQ_LEN"
      --single-stride "$SINGLE_STRIDE"
      --single-min-sequences "$SINGLE_MIN_SEQUENCES"
      --single-seq-window-stride "$SINGLE_SEQ_WINDOW_STRIDE"
      --low-resource-threshold "$LOW_RESOURCE_THRESHOLD"
    )
  fi

  echo ""
  echo "=== Converting: dataset=$dataset_type scene=$scene ==="
  "${cmd[@]}"
}

main() {
  cd "$(dirname "$0")/.."

  for scene in $SCENES; do
    if [[ "$DATASET_MODE" == "single" || "$DATASET_MODE" == "both" ]]; then
      run_convert "single" "$scene"
    fi
    if [[ "$DATASET_MODE" == "equirect" || "$DATASET_MODE" == "both" ]]; then
      run_convert "equirect" "$scene"
    fi
  done

  echo ""
  echo "Done converting all requested scenes."
}

main
