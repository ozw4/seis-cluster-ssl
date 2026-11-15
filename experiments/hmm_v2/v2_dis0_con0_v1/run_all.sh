#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export SEIS_SSL_CLUSTER_WORKSPACE="$PWD"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?Set the existing artifact root}"
: "${F3_ROOT:?Preserve the frozen F3 label path}"
: "${SEIS_SSL_CLUSTER_VOLVE_ROOT:?Set the existing Volve data root}"
read -r -a devices <<< "${V2_DIS0_CON0_GPUS:-0 1}"
python -m seis_ssl_cluster.hmm.v2_dis0_con0 --phase run --gpus "${devices[@]}"
