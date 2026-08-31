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

models=(
	mae
	mae_hmm_k6
	local_barlow_twins
	local_barlow_twins_hmm_k6
	random
)
layouts=(layout_000 layout_001 layout_002 layout_003 layout_004)
sizes=(small medium large)

run_or_list() {
	if [[ "${LIST_ONLY}" == '1' ]]; then
		printf '%q' "$1"
		shift
		printf ' %q' "$@"
		printf '\n'
		return
	fi
	"$@"
}

preflight=(
	"${PYTHON_BIN}"
	"${REPO_ROOT}/proc/seis_ssl_cluster/audit_volve_horizon_five_way_sources.py"
	--config "${CONFIG}"
)
if [[ "${LIST_ONLY}" == '1' ]]; then
	preflight+=(--dry-run)
fi
run_or_list "${preflight[@]}"

for model in "${models[@]}"; do
	for layout in "${layouts[@]}"; do
		for size in "${sizes[@]}"; do
			command=(
				"${PYTHON_BIN}"
				"${REPO_ROOT}/proc/seis_ssl_cluster/run_volve_horizon_five_way.py"
				--config "${CONFIG}"
				--model "${model}"
				--layout "${layout}"
				--size "${size}"
				--layout-config "${LAYOUT_CONFIG}"
				--device "${DEVICE_NAME}"
			)
			if [[ "${LIST_ONLY}" == '1' ]]; then
				command+=(--dry-run)
			fi
			run_or_list "${command[@]}"
		done
	done
done
