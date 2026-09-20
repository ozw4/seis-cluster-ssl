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
  : "${F3_ROOT:?Set F3 data root}"
  : "${SEIS_SSL_CLUSTER_VOLVE_ROOT:?Set Volve data root}"
  python -m seis_ssl_cluster.hmm.multi_source_receipts --check \
    --matrix "$study_dir/matrix.yaml" \
    --controls "$repository_dir/experiments/hmm_v2/k6810_multi_source_evaluation_v1/hmm_v1_controls_receipt.json" \
    --reuse "$study_dir/f3_mae_k468_reuse_receipt.json" \
    --artifact-root "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT"
fi
for experiment in \
  experiments/f3/facies_benchmark_v2/129_hmm_v2_k468_multi_source_v1 \
  experiments/parihaka/facies_benchmark_v1/42_hmm_v2_k468_multi_source_v1 \
  experiments/volve/horizon_benchmark_v1/35_hmm_v2_k468_multi_source_v1; do
  if [[ "$mode" == --dry-run ]]; then
    # Later stages require outputs from earlier stages; validate existing target inputs only.
    bash "$experiment/run_all.sh" --dry-run --from targets --to targets
  else
    bash "$experiment/run_all.sh" "$mode"
  fi
done
if [[ "$mode" == --execute ]]; then
  python -m seis_ssl_cluster.hmm.multi_source_aggregate --config "$study_dir/aggregate.yaml"
elif [[ "$mode" == --plan ]]; then
  printf 'After all surveys: python -m seis_ssl_cluster.hmm.multi_source_aggregate --config %q\n' "$study_dir/aggregate.yaml"
fi
