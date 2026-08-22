#!/usr/bin/env bash
set -euo pipefail

python proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py \
  --clustering-output-dir "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/clustering/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/hmm_targets/local_bt100/k6" \
  --pseudo-target-root "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pseudo_targets/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/local_bt100" \
  --k 6 \
  --confidence 1.0 \
  --boundary-alpha 0.0 \
  --boundary-tau 1.0 \
  --schema-version 2
