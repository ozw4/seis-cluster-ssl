#!/usr/bin/env bash
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then exit 2; fi
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?Set artifact root}"
python proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py --clustering-output-dir "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/clustering/f3/facies_benchmark_v1/hmm_v2_k6810_multi_source_v1/random/k8k10" --pseudo-target-root "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pseudo_targets/f3/facies_benchmark_v1/hmm_v2_k6810_multi_source_v1/random/shared" --k 8 --confidence 1.0 --boundary-alpha 0.0 --boundary-tau 1.0 --schema-version 2 "$@"
python proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py --clustering-output-dir "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/clustering/f3/facies_benchmark_v1/hmm_v2_k6810_multi_source_v1/random/k8k10" --pseudo-target-root "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pseudo_targets/f3/facies_benchmark_v1/hmm_v2_k6810_multi_source_v1/random/shared" --k 10 --confidence 1.0 --boundary-alpha 0.0 --boundary-tau 1.0 --schema-version 2 "$@"
