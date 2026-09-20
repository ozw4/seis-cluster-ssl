#!/usr/bin/env bash
set -euo pipefail
study_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_dir="$(cd "$study_dir/../../.." && pwd)"
cd "$repository_dir"
export SEIS_SSL_CLUSTER_WORKSPACE="$repository_dir"
export PYTHONPATH="$repository_dir/src${PYTHONPATH:+:$PYTHONPATH}"
mode="${1:---plan}"
if [[ $# -gt 1 || ( "$mode" != --plan && "$mode" != --dry-run && "$mode" != --execute ) ]]; then
  echo 'Usage: run_all.sh [--plan|--dry-run|--execute]' >&2
  exit 2
fi
if [[ "$mode" != --plan ]]; then
  : "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?Set artifact root}"
  : "${F3_ROOT:?Set frozen F3 supervision data root}"
  : "${SEIS_SSL_CLUSTER_VOLVE_ROOT:?Set Volve data root}"
fi
for experiment in experiments/f3/facies_benchmark_v2/130_hmm_v2_k468_cons010_multi_source_v1 experiments/parihaka/facies_benchmark_v1/43_hmm_v2_k468_cons010_multi_source_v1 experiments/volve/horizon_benchmark_v1/36_hmm_v2_k468_cons010_multi_source_v1; do
  if [[ "$mode" == --dry-run ]]; then
    bash "$experiment/run_all.sh" --dry-run --from smoke --to full
  else
    bash "$experiment/run_all.sh" "$mode" --from smoke
  fi
done
if [[ "$mode" == --execute ]]; then
  python -m seis_ssl_cluster.hmm.multi_source_aggregate --config "$study_dir/aggregate.yaml"
elif [[ "$mode" == --plan ]]; then
  printf 'After all surveys: python -m seis_ssl_cluster.hmm.multi_source_aggregate --config %q\n' "$study_dir/aggregate.yaml"
fi
