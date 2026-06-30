#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Benchmark runner for retrieval-evaluation.
# Runs hloc-megaloc and seqfg by default. Set RUN_METHODS to include
# temporal-smooth or other supported methods.
#
# Usage:
#   chmod +x bench_server.sh
#   QUERY_SETS="query_val_phone query_val_hololens" ./bench_server.sh

WORK_ROOT="${WORK_ROOT:-$PWD}"
SOURCE_ROOT="${SOURCE_ROOT:-$WORK_ROOT/../../../code}"
DATASET_ROOT="${DATASET_ROOT:-$WORK_ROOT/../../../datasets}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-$WORK_ROOT/../../../outputs2}"

REPO_DIR="${REPO_DIR:-$SOURCE_ROOT/retrieval-evaluation}"
DATASET="${DATASET:-lamar}"
SCENE="${SCENE:-HGE}"
MAP_DIRS="${MAP_DIRS:-${MAP_DIR:-map}}"
QUERY_SETS="${QUERY_SETS:-query_val_phone}"
THRESHOLD_M="${THRESHOLD_M:-5.0}"

VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
CONDA_ENV="${CONDA_ENV:-vnu-ar}"
GLOBAL_RUN_ID="${GLOBAL_RUN_ID:-}"  # Will be set by runner; if empty, synthesize locally

RETRIEVAL_CONFIG_HLOC="${RETRIEVAL_CONFIG_HLOC:-configs/retrieval.hloc.yaml}"
RETRIEVAL_CONFIG_SEQFG="${RETRIEVAL_CONFIG_SEQFG:-configs/retrieval.sequence.yaml}"
RETRIEVAL_CONFIG_TEMPORAL="${RETRIEVAL_CONFIG_TEMPORAL:-configs/retrieval.temporal_smooth.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/eval.yaml}"
RUN_METHODS="${RUN_METHODS:-hloc-megaloc seqfg}"

declare -a MAP_DIR_LIST=()
declare -a QUERY_SET_LIST=()
declare -a METHOD_LIST=()

log() {
  echo "[$(date +'%F %T')] $*"
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

on_error() {
  local line="$1"
  local exit_code="$2"
  echo "[ERROR] Failed at line ${line} (exit code: ${exit_code})" >&2
}
trap 'on_error $LINENO $?' ERR

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "Missing file: $path"
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || die "Missing directory: $path"
}

build_global_run_id() {
  # Fallback if GLOBAL_RUN_ID not provided by runner.
  local git_sha
  git_sha="$(git -C "$REPO_DIR" rev-parse --short HEAD)"
  echo "bench_$(date +%Y%m%d_%H%M%S)_${git_sha}"
}

print_artifact_paths() {
  local method="$1"
  local run_id="$2"
  local map_dir="$3"
  for q in "${QUERY_SET_LIST[@]}"; do
    local root="$OUTPUTS_ROOT/$run_id/$DATASET/$SCENE/$map_dir/$q/$method"
    echo ""
    echo "[ARTIFACT] map_dir=$map_dir query_set=$q method=$method"
    echo "  - $root/retrieval/pairs.txt"
    echo "  - $root/eval/evaluation.json"
    echo "  - $root/eval/per_query.jsonl"
    echo "  - $root/meta/stage_status.json"
  done
}

run_benchmark_method() {
  local method="$1"
  local config="$2"
  local map_dir="$3"
  local query_sets_str
  local IFS=' '

  query_sets_str="${QUERY_SET_LIST[*]}"

  log "Running benchmark: map_dir=$map_dir method=$method run_id=$GLOBAL_RUN_ID"
  cd "$REPO_DIR"
  # shellcheck disable=SC1090
  # source "$VENV_DIR/bin/activate"

  conda run -n "$CONDA_ENV" \
  env DATASET_ROOT="$DATASET_ROOT" \
  DATASET="$DATASET" \
  SCENE="$SCENE" \
  MAP_DIR="$map_dir" \
  QUERY_SETS="$query_sets_str" \
  OUTPUTS_ROOT="$OUTPUTS_ROOT" \
  RETRIEVAL_METHOD="$method" \
  RUN_ID="$GLOBAL_RUN_ID" \
  THRESHOLD_M="$THRESHOLD_M" \
  RETRIEVAL_CONFIG="$config" \
  EVAL_CONFIG="$EVAL_CONFIG" \
  MODE=run \
  PYTHON_BIN="python" \
  ./run.sh

  print_artifact_paths "$method" "$GLOBAL_RUN_ID" "$map_dir"
}

config_for_method() {
  local method="$1"
  case "$method" in
    hloc-megaloc)
      echo "$RETRIEVAL_CONFIG_HLOC"
      ;;
    seqfg|seqprob)
      echo "$RETRIEVAL_CONFIG_SEQFG"
      ;;
    temporal-smooth)
      echo "$RETRIEVAL_CONFIG_TEMPORAL"
      ;;
    *)
      die "No config mapping for method: $method"
      ;;
  esac
}

parse_space_list() {
  local input="$1"
  local -n output_ref="$2"

  # Split on spaces to avoid relying on global IFS.
  local IFS=' '
  read -r -a output_ref <<< "$input"
}

main() {
  log "Starting benchmark runner..."

  require_dir "$REPO_DIR"
  require_file "$REPO_DIR/run.sh"
  # require_file "$VENV_DIR/bin/activate"
  require_file "$REPO_DIR/$RETRIEVAL_CONFIG_HLOC"
  require_file "$REPO_DIR/$RETRIEVAL_CONFIG_SEQFG"
  require_file "$REPO_DIR/$RETRIEVAL_CONFIG_TEMPORAL"
  require_file "$REPO_DIR/$EVAL_CONFIG"

  local scene_root="$DATASET_ROOT/$DATASET/$SCENE"
  require_dir "$scene_root"
  parse_space_list "$MAP_DIRS" MAP_DIR_LIST
  parse_space_list "$QUERY_SETS" QUERY_SET_LIST
  parse_space_list "$RUN_METHODS" METHOD_LIST

  [[ ${#MAP_DIR_LIST[@]} -gt 0 ]] || die "MAP_DIRS is empty"
  [[ ${#QUERY_SET_LIST[@]} -gt 0 ]] || die "QUERY_SETS is empty"
  [[ ${#METHOD_LIST[@]} -gt 0 ]] || die "RUN_METHODS is empty"

  for map_dir in "${MAP_DIR_LIST[@]}"; do
    require_dir "$scene_root/$map_dir/images"
    require_file "$scene_root/$map_dir/images.txt"

    for q in "${QUERY_SET_LIST[@]}"; do
      require_dir "$scene_root/$q/images"
      if [[ ! -f "$scene_root/$q/images.txt" && ! -f "$scene_root/$q/query_names.txt" ]]; then
        die "Missing images.txt/query_names.txt for query set: $scene_root/$q"
      fi
    done
  done

  log "Resolved maps: ${MAP_DIR_LIST[*]}"
  log "Resolved query sets: ${QUERY_SET_LIST[*]}"
  log "Resolved methods: ${METHOD_LIST[*]}"

  # Use provided GLOBAL_RUN_ID or synthesize if missing.
  if [[ -z "$GLOBAL_RUN_ID" ]]; then
    GLOBAL_RUN_ID="$(build_global_run_id)"
    log "GLOBAL_RUN_ID not provided; synthesized: $GLOBAL_RUN_ID"
  fi

  for map_dir in "${MAP_DIR_LIST[@]}"; do
    local method
    for method in "${METHOD_LIST[@]}"; do
      run_benchmark_method "$method" "$(config_for_method "$method")" "$map_dir"
    done
  done

  log "Done. Benchmark completed."
}

main "$@"
