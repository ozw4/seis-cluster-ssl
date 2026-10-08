#!/usr/bin/env bash
set -euo pipefail
experiment_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd "${experiment_root}/../../../.." && pwd)"
cd "${workspace_root}"
exec "${SEIS_SSL_CLUSTER_PYTHON:-python}" proc/seis_ssl_cluster/run_hmm_v3_trace_mask_lr3.py \
  --experiment "${experiment_root}" "$@"

