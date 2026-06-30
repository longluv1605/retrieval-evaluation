#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Minimal full-matrix benchmark runner.
# Runs baseline hloc-megaloc and seqfg for:
# dataset x variant x scene x map x query

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-$REPO_DIR/../../datasets}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-$REPO_DIR/../../outputs2}"
BENCH_SERVER="${BENCH_SERVER:-$SCRIPT_DIR/bench_server.sh}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-$REPO_DIR/tools/build_dataset_summary.py}"
SUMMARY_JSON="${SUMMARY_JSON:-$REPO_DIR/../../resources/benchmarks/dataset_summary.generated.json}"
MATRIX_SCRIPT="${MATRIX_SCRIPT:-$SCRIPT_DIR/build_job_matrix.py}"
AGGREGATE_SCRIPT="${AGGREGATE_SCRIPT:-$SCRIPT_DIR/aggregate_recall_table.py}"

SEQFG_CONFIG="${SEQFG_CONFIG:-$REPO_DIR/configs/retrieval.sequence.yaml}"
BASE_HLOC_CONFIG="${BASE_HLOC_CONFIG:-$REPO_DIR/configs/retrieval.hloc.yaml}"
TEMPORAL_CONFIG="${TEMPORAL_CONFIG:-$REPO_DIR/configs/retrieval.temporal_smooth.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-$REPO_DIR/configs/eval.yaml}"
RUN_METHODS="${RUN_METHODS:-hloc-megaloc temporal-smooth seqfg}"

CONDA_ENV="${CONDA_ENV:-vnu-ar}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-fullbench}"
THRESHOLD_M="${THRESHOLD_M:-1.0}"
DRY_RUN="${DRY_RUN:-0}"

# Generate global run_id once for the entire benchmark session unless provided.
GLOBAL_RUN_ID="${GLOBAL_RUN_ID:-${RUN_ID_PREFIX}_$(date +%Y%m%d_%H%M%S)_$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)}"

log() {
  echo "[$(date +'%F %T')] $*"
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "Missing file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "Missing directory: $1"
}

to_repo_rel() {
  local p="$1"
  python3 - <<'PY' "$REPO_DIR" "$p"
from pathlib import Path
import sys

repo = Path(sys.argv[1]).resolve()
p = Path(sys.argv[2]).resolve()
try:
    print(p.relative_to(repo))
except ValueError:
    print(f"[ERROR] Path must be inside repo: {p} (repo: {repo})", file=sys.stderr)
    raise SystemExit(2)
PY
}

require_file "$BENCH_SERVER"
require_file "$SUMMARY_SCRIPT"
require_file "$MATRIX_SCRIPT"
require_file "$AGGREGATE_SCRIPT"
require_file "$SEQFG_CONFIG"
require_file "$BASE_HLOC_CONFIG"
require_file "$TEMPORAL_CONFIG"
require_file "$EVAL_CONFIG"
require_dir "$DATASET_ROOT"

SEQFG_CONFIG_REL="$(to_repo_rel "$SEQFG_CONFIG")"
BASE_HLOC_CONFIG_REL="$(to_repo_rel "$BASE_HLOC_CONFIG")"
TEMPORAL_CONFIG_REL="$(to_repo_rel "$TEMPORAL_CONFIG")"
EVAL_CONFIG_REL="$(to_repo_rel "$EVAL_CONFIG")"

log "Generating dataset summary from filesystem..."
python3 "$SUMMARY_SCRIPT" \
  --dataset-root "$DATASET_ROOT" \
  --output "$SUMMARY_JSON"

log "Building job matrix from summary..."
JOB_LINES="$(python3 "$MATRIX_SCRIPT" --summary "$SUMMARY_JSON")"
[[ -n "$JOB_LINES" ]] || die "No jobs found in summary: $SUMMARY_JSON"

TOTAL_JOBS=$(printf '%s\n' "$JOB_LINES" | wc -l | tr -d ' ')
log "Jobs: $TOTAL_JOBS, SeqFG config: $SEQFG_CONFIG"
log "Run methods: $RUN_METHODS"
log "Global run_id: $GLOBAL_RUN_ID"

run_one() {
  local dataset="$1"
  local scene="$2"
  local map_dir="$3"
  local query_set="$4"

  log "Run dataset=$dataset scene=$scene map=$map_dir query=$query_set"

  if [[ "$DRY_RUN" == "1" ]]; then
    return
  fi

  (
    cd "$(dirname "$BENCH_SERVER")"
    CONDA_ENV="$CONDA_ENV" \
    DATASET_ROOT="$DATASET_ROOT" \
    OUTPUTS_ROOT="$OUTPUTS_ROOT" \
    DATASET="$dataset" \
    SCENE="$scene" \
    MAP_DIRS="$map_dir" \
    QUERY_SETS="$query_set" \
    THRESHOLD_M="$THRESHOLD_M" \
    GLOBAL_RUN_ID="$GLOBAL_RUN_ID" \
    RETRIEVAL_CONFIG_HLOC="$BASE_HLOC_CONFIG_REL" \
    RETRIEVAL_CONFIG_SEQFG="$SEQFG_CONFIG_REL" \
    RETRIEVAL_CONFIG_TEMPORAL="$TEMPORAL_CONFIG_REL" \
    EVAL_CONFIG="$EVAL_CONFIG_REL" \
    RUN_METHODS="$RUN_METHODS" \
    bash "$BENCH_SERVER"
  )
}

while IFS='|' read -r DATASET SCENE MAP_DIR QUERY_SET; do
  run_one "$DATASET" "$SCENE" "$MAP_DIR" "$QUERY_SET"
done <<< "$JOB_LINES"

if [[ "$DRY_RUN" != "1" ]]; then
  AGG_CSV="$OUTPUTS_ROOT/$GLOBAL_RUN_ID/summary_recall.csv"
  AGG_MD="$OUTPUTS_ROOT/$GLOBAL_RUN_ID/summary_recall.md"

  log "Aggregating recall table..."
  python3 "$AGGREGATE_SCRIPT" \
    --outputs-root "$OUTPUTS_ROOT" \
    --run-id "$GLOBAL_RUN_ID" \
    --out-csv "$AGG_CSV" \
    --out-md "$AGG_MD"

  log "Summary CSV: $AGG_CSV"
  log "Summary MD: $AGG_MD"
fi

log "Done."
