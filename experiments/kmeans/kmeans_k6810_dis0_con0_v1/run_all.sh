#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${KMEANS_ZERO_GPU:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

exec python -u proc/seis_ssl_cluster/run_kmeans_study.py \
  --matrix experiments/kmeans/kmeans_k6810_dis0_con0_v1/matrix.yaml \
  --execute --from smoke
