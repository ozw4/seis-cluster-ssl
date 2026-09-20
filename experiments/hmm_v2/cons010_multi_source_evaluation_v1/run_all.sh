#!/usr/bin/env bash
set -euo pipefail
study_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="${1:---plan}"
if [[ $# -gt 1 || ( "$mode" != --plan && "$mode" != --dry-run && "$mode" != --execute ) ]]; then
  echo 'Usage: run_all.sh [--plan|--dry-run|--execute]' >&2
  exit 2
fi
k468="$study_dir/../k468_cons010_multi_source_evaluation_v1/run_all.sh"
k6810="$study_dir/../k6810_cons010_multi_source_evaluation_v1/run_all.sh"
if [[ "$mode" != --execute ]]; then
  bash "$k468" "$mode"
  bash "$k6810" "$mode"
  exit 0
fi
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?Set artifact root}"
operation="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/operations/hmm_v2/cons010_multi_source_evaluation_v1"
mkdir -p "$operation"
exec 9> "$operation/launcher.lock"
flock --nonblock 9
gpu_k468="${K468_GPU:-0}"
gpu_k6810="${K6810_GPU:-1}"
if [[ "$gpu_k468" == "$gpu_k6810" ]]; then
  echo 'The parallel launcher requires two distinct GPU assignments.' >&2
  exit 2
fi
invocation="$(mktemp -d "$operation/invocation-XXXXXXXX")"
printf 'Logs: %s\nK468 GPU: %s\nK6810 GPU: %s\n' "$invocation" "$gpu_k468" "$gpu_k6810"
CUDA_VISIBLE_DEVICES="$gpu_k468" bash "$k468" --execute > "$invocation/k468.log" 2>&1 &
pid_k468=$!
CUDA_VISIBLE_DEVICES="$gpu_k6810" bash "$k6810" --execute > "$invocation/k6810.log" 2>&1 &
pid_k6810=$!
status_k468=0
status_k6810=0
wait "$pid_k468" || status_k468=$?
wait "$pid_k6810" || status_k6810=$?
printf 'k468=%s\nk6810=%s\n' "$status_k468" "$status_k6810" > "$invocation/exit_status.txt"
[[ "$status_k468" == 0 && "$status_k6810" == 0 ]]
