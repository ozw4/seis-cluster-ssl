#!/usr/bin/env bash
set -euo pipefail

python proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py \
  --clustering-output-dir "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/clustering/f3/facies_benchmark_v1/random_init_self_target_hmm_k6_distill010_v1/hmm_targets/random_init/k6" \
  --pseudo-target-root "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pseudo_targets/f3/facies_benchmark_v1/random_init_self_target_hmm_k6_distill010_v1/random_init" \
  --k 6 \
  --confidence 1.0 \
  --boundary-alpha 0.0 \
  --boundary-tau 1.0 \
  --schema-version 2 \
  "$@"
