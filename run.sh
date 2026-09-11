#!/usr/bin/env bash
set -euo pipefail

# Retrieval-evaluation launcher (LAMAR-style env overrides)
# Usage:
#   chmod +x run.sh
#   ./run.sh
#   RETRIEVAL_METHOD=hloc-megaloc QUERY_SETS="query_val_phone query_val_hololens" ./run.sh
#
# Modes:
#   MODE=run                  : retrieval + eval-distance
#   MODE=retrieve             : retrieval only
#   MODE=eval                 : eval-distance only

DATASET_ROOT="${DATASET_ROOT:-../datasets}"
DATASET="${DATASET:-lamar}"
SCENE="${SCENE:-HGE}"
MAP_DIR="${MAP_DIR:-map}"
QUERY_SETS="${QUERY_SETS:-query_val_phone}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-../outputs}"
RUN_ID="${RUN_ID:-exp_001}"
RETRIEVAL_METHOD="${RETRIEVAL_METHOD:-seqfg}"
THRESHOLD_M="${THRESHOLD_M:-10.0}"

RETRIEVAL_CONFIG="${RETRIEVAL_CONFIG:-configs/retrieval.sequence.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/eval.yaml}"

MODE="${MODE:-run}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d "$DATASET_ROOT" ]]; then
	echo "[ERROR] DATASET_ROOT not found: $DATASET_ROOT"
	exit 1
fi

if [[ ! -f "$RETRIEVAL_CONFIG" ]]; then
	echo "[ERROR] RETRIEVAL_CONFIG not found: $RETRIEVAL_CONFIG"
	exit 1
fi

if [[ ! -f "$EVAL_CONFIG" ]]; then
	echo "[ERROR] EVAL_CONFIG not found: $EVAL_CONFIG"
	exit 1
fi

echo "Running retrieval-evaluation with parameters:"
echo "  DATASET_ROOT:      $DATASET_ROOT"
echo "  DATASET:           $DATASET"
echo "  SCENE:             $SCENE"
echo "  MAP_DIR:           $MAP_DIR"
echo "  QUERY_SETS:        $QUERY_SETS"
echo "  OUTPUTS_ROOT:      $OUTPUTS_ROOT"
echo "  RETRIEVAL_METHOD:  $RETRIEVAL_METHOD"
echo "  RUN_ID:            $RUN_ID"
echo "  THRESHOLD_M:       $THRESHOLD_M"
echo "  RETRIEVAL_CONFIG:  $RETRIEVAL_CONFIG"
echo "  EVAL_CONFIG:       $EVAL_CONFIG"
echo "  MODE:              $MODE"
echo "  PYTHON_BIN:        $PYTHON_BIN"

for QUERY_SET in $QUERY_SETS; do
	echo ""
	echo "[INFO] Processing query set: $QUERY_SET"

	if [[ "$MODE" == "retrieve" ]]; then
		"$PYTHON_BIN" -m retrieval_eval.cli retrieve \
			--config "$RETRIEVAL_CONFIG" \
			--dataset-root "$DATASET_ROOT" \
			--dataset "$DATASET" \
			--scene "$SCENE" \
			--map-dir "$MAP_DIR" \
			--query-set "$QUERY_SET" \
			--outputs-root "$OUTPUTS_ROOT" \
			--retrieval-method "$RETRIEVAL_METHOD" \
			--run-id "$RUN_ID"
	elif [[ "$MODE" == "eval" ]]; then
		"$PYTHON_BIN" -m retrieval_eval.cli eval-distance \
			--config "$EVAL_CONFIG" \
			--dataset-root "$DATASET_ROOT" \
			--dataset "$DATASET" \
			--scene "$SCENE" \
			--map-dir "$MAP_DIR" \
			--query-set "$QUERY_SET" \
			--outputs-root "$OUTPUTS_ROOT" \
			--retrieval-method "$RETRIEVAL_METHOD" \
			--run-id "$RUN_ID" \
			--threshold-m "$THRESHOLD_M"
	elif [[ "$MODE" == "run" ]]; then
		"$PYTHON_BIN" -m retrieval_eval.cli run \
			--retrieval-config "$RETRIEVAL_CONFIG" \
			--eval-config "$EVAL_CONFIG" \
			--dataset-root "$DATASET_ROOT" \
			--dataset "$DATASET" \
			--scene "$SCENE" \
			--map-dir "$MAP_DIR" \
			--query-set "$QUERY_SET" \
			--outputs-root "$OUTPUTS_ROOT" \
			--retrieval-method "$RETRIEVAL_METHOD" \
			--run-id "$RUN_ID" \
			--threshold-m "$THRESHOLD_M"
	else
		echo "[ERROR] Unsupported MODE: $MODE (allowed: run, retrieve, eval)"
		exit 1
	fi
done

echo ""
echo "[DONE] Finished MODE=$MODE for QUERY_SETS: $QUERY_SETS"
