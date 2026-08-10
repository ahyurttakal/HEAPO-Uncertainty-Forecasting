#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/local.yaml}"
RUN_DIR="${2:-outputs/heapo_main}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m heapo_forecasting.cli full --config "${CONFIG_PATH}"
"${PYTHON_BIN}" -m heapo_forecasting.cli finalize --config "${CONFIG_PATH}"
"${PYTHON_BIN}" scripts/make_manuscript_outputs.py \
  --run-dir "${RUN_DIR}" --dpi 600
