#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CONFIG="${SCRIPT_DIR}/50_five_way.yaml"
LAYOUT_CONFIG="${REPO_ROOT}/experiments/volve/horizon_benchmark_v1/20_horizon_supervision/01_layouts.yaml"
PYTHON_BIN="${PYTHON:-python}"
DEVICE_NAME="${DEVICE:-cuda}"
LIST_ONLY="${DRY_RUN:-0}"

if [[ "${LIST_ONLY}" != '0' && "${LIST_ONLY}" != '1' ]]; then
	printf 'DRY_RUN must be 0 or 1, got %q\n' "${LIST_ONLY}" >&2
	exit 2
fi

command=(
	"${PYTHON_BIN}"
	"${REPO_ROOT}/proc/seis_ssl_cluster/run_volve_horizon_five_way_suite.py"
	--config "${CONFIG}"
	--layout-config "${LAYOUT_CONFIG}"
	--device "${DEVICE_NAME}"
)
if [[ "${LIST_ONLY}" == '1' ]]; then
	command+=(--dry-run)
fi
"${command[@]}" "$@"
