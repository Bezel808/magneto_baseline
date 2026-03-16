#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zongze/mengshichen_projects/magneto_baseline"
PYTHON_BIN="${PYTHON_BIN:-/home/zongze/.venvs/magneto/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/home/mengshi/table_quality/datasets_joint_discovery_integration}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/runs/magneto_sm_1218}"
MODE="${MODE:-header_values_default}"
TOPK="${TOPK:-20}"
EMB_THRESH="${EMB_THRESH:-0.1}"
DEVICE="${DEVICE:-cpu}"
MAX_PAIRS_PER_WORKER="${MAX_PAIRS_PER_WORKER:-30}"
DATASETS="${DATASETS:-wikidbs_1218 santos_benchmark_1218 magellan_1218}"

cd "${ROOT}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="${ROOT}/algorithms/magneto:${PYTHONPATH:-}"

echo "[RUN] ROOT=${ROOT}"
echo "[RUN] DATASETS=${DATASETS}"
echo "[RUN] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[RUN] DEVICE=${DEVICE}"

"${PYTHON_BIN}" run_magneto_sm_1218_chunked.py \
  --dataset-root "${DATASET_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --mode "${MODE}" \
  --topk "${TOPK}" \
  --embedding-threshold "${EMB_THRESH}" \
  --device "${DEVICE}" \
  --max-pairs-per-worker "${MAX_PAIRS_PER_WORKER}" \
  --datasets ${DATASETS} \
  "$@"
