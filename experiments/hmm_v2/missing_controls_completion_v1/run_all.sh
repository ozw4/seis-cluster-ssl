#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export SEIS_SSL_CLUSTER_WORKSPACE="$PWD"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?Set the existing artifact root}"
: "${F3_ROOT:?Preserve the frozen F3 label path}"
: "${SEIS_SSL_CLUSTER_VOLVE_ROOT:?Set the existing Volve root}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m seis_ssl_cluster.hmm.missing_controls --phase run --gpu "${MISSING_CONTROLS_GPU:-0}"
