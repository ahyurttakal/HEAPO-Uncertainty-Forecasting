#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/local.yaml}"
RUN_DIR="${2:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m heapo_forecasting.cli full --config "${CONFIG_PATH}"
"${PYTHON_BIN}" -m heapo_forecasting.cli finalize --config "${CONFIG_PATH}"
if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$("${PYTHON_BIN}" -c 'from heapo_forecasting.config import load_config, run_dir; import sys; print(run_dir(load_config(sys.argv[1])))' "${CONFIG_PATH}")"
fi
"${PYTHON_BIN}" scripts/make_manuscript_outputs.py \
  --run-dir "${RUN_DIR}" --dpi 600 --tiff
