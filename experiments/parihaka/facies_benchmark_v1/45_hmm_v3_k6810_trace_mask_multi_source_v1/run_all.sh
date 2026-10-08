#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SEIS_SSL_CLUSTER_WORKSPACE="${SEIS_SSL_CLUSTER_WORKSPACE:-$(cd -- "$experiment_dir/../../../.." && pwd)}"
cd -- "$SEIS_SSL_CLUSTER_WORKSPACE"
exec "${SEIS_SSL_CLUSTER_PYTHON:-python}" -m seis_ssl_cluster.hmm.trace_mask_driver \
  --experiment "$experiment_dir" "$@"
