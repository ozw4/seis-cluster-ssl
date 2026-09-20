#!/usr/bin/env bash
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then exit 2; fi
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?Set artifact root}"
python proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py --clustering-output-dir "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/surveys/volve/horizon_benchmark_v1/clustering/hmm_v2_k468_multi_source_v1/local_bt3/k4k8" --pseudo-target-root "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/surveys/volve/horizon_benchmark_v1/pseudo_targets/hmm_v2_k468_multi_source_v1/local_bt3/shared" --k 4 --confidence 1.0 --boundary-alpha 0.0 --boundary-tau 1.0 --schema-version 2 "$@"
python proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py --clustering-output-dir "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/surveys/volve/horizon_benchmark_v1/clustering/hmm_v2_k468_multi_source_v1/local_bt3/k4k8" --pseudo-target-root "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/surveys/volve/horizon_benchmark_v1/pseudo_targets/hmm_v2_k468_multi_source_v1/local_bt3/shared" --k 8 --confidence 1.0 --boundary-alpha 0.0 --boundary-tau 1.0 --schema-version 2 "$@"
