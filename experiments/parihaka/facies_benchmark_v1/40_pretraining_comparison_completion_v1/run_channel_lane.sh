#!/usr/bin/env bash
set -euo pipefail
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?Set the shared artifact root}"
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1
exec python -m seis_ssl_cluster.parihaka.channel_lane "$@"
