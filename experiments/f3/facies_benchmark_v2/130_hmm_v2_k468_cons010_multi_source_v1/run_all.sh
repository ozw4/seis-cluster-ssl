#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_dir="$(cd "$experiment_dir/../../../.." && pwd)"
cd "$repository_dir"
export SEIS_SSL_CLUSTER_WORKSPACE="$repository_dir"
export PYTHONPATH="$repository_dir/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m seis_ssl_cluster.hmm.multi_source_driver --experiment "$experiment_dir" "$@"
