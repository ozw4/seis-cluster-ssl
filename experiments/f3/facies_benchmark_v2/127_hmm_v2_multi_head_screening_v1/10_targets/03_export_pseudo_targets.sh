#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?Set the existing artifact root}"

for k in 4 8 10 6; do
  source_suffix=k4810
  target_suffix=shared
  if [[ "$k" == 6 ]]; then
    source_suffix=k6_replay
    target_suffix=k6_replay
  fi
  python proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py \
    --clustering-output-dir "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/clustering/f3/facies_benchmark_v1/hmm_v2_multi_head_screening_v1/mae100/${source_suffix}" \
    --pseudo-target-root "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pseudo_targets/f3/facies_benchmark_v1/hmm_v2_multi_head_screening_v1/mae100/${target_suffix}" \
    --k "$k" \
    --confidence 1.0 \
    --boundary-alpha 0.0 \
    --boundary-tau 1.0 \
    --schema-version 2 \
    "$@"
done
