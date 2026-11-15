#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export SEIS_SSL_CLUSTER_WORKSPACE="$PWD"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?Set the existing artifact root}"
: "${F3_ROOT:?Use the frozen F3 label path}"
: "${SEIS_SSL_CLUSTER_VOLVE_ROOT:?Set Volve data root}"
read -r -a devices <<< "${SENSITIVITY_GPUS:-${SENSITIVITY_GPU:-1}}"
operations="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/operations/hmm_v2/distillation_sensitivity_v1"
mkdir -p "$operations"
exec 9>"$operations/study.lock"
flock -n 9
jobs=(f3:0.0 f3:0.1 f3:0.4 parihaka:0.0 parihaka:0.1 parihaka:0.4 volve:0.0 volve:0.1 volve:0.4)
run() {
  python -m seis_ssl_cluster.hmm.distillation_sensitivity "$@"
}
run_phase() {
  local phase="$1" slot index gpu pid status=0
  local slot_count=$((2 * ${#devices[@]}))
  local pids=()
  for ((slot=0; slot<slot_count; slot++)); do
    (
      gpu="${devices[slot / 2]}"
      export CUDA_VISIBLE_DEVICES="$gpu"
      for ((index=slot; index<${#jobs[@]}; index+=slot_count)); do
        run --phase "$phase" --survey "${jobs[index]%:*}" --weight "${jobs[index]#*:}" --resume
      done
    ) &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  return "$status"
}
run --phase validate
# Each selected GPU has two fixed lanes. Every phase has an all-condition barrier.
for phase in full embeddings medium remaining; do
  run_phase "$phase"
done
run --phase test --resume
run --phase summary
python -m seis_ssl_cluster.hmm.distillation_review
