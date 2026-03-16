#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-${SCRIPT_DIR}}"

# Python resolution priority:
# 1) user-provided PYTHON_BIN
# 2) $HOME virtualenv magneto python
# 3) system python3
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [[ -x "${HOME}/.venvs/magneto/bin/python" ]]; then
  PYTHON_BIN="${HOME}/.venvs/magneto/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

DATASET_ROOT="${DATASET_ROOT:-/home/mengshi/table_quality/datasets_joint_discovery_integration}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/runs/magneto_sm_1218}"
MODE="${MODE:-header_values_default}"
TOPK="${TOPK:-20}"
EMB_THRESH="${EMB_THRESH:-0.1}"
DEVICE="${DEVICE:-cpu}"
MAX_PAIRS_PER_WORKER="${MAX_PAIRS_PER_WORKER:-30}"
DATASETS="${DATASETS:-wikidbs_1218 santos_benchmark_1218 magellan_1218}"

cd "${ROOT}"
if [[ ! -f "${ROOT}/run_magneto_sm_1218_chunked.py" ]]; then
  echo "[ERROR] Missing runner: ${ROOT}/run_magneto_sm_1218_chunked.py" >&2
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Python not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="${ROOT}/algorithms/magneto:${PYTHONPATH:-}"

echo "[RUN] ROOT=${ROOT}"
echo "[RUN] PYTHON_BIN=${PYTHON_BIN}"
echo "[RUN] DATASET_ROOT=${DATASET_ROOT}"
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
