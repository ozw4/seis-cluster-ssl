#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export SEIS_SSL_CLUSTER_WORKSPACE="$PWD"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?Set the existing artifact root}"
: "${F3_ROOT:?Use the frozen F3 label path}"
: "${SEIS_SSL_CLUSTER_VOLVE_ROOT:?Set the existing Volve root}"
main_gpu="${HMM_V1_MAIN_GPU:-0}"
volve_gpu="${HMM_V1_VOLVE_GPU:-1}"
operations="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/operations/hmm_v1/zero_distillation_ablation_v1"
mkdir -p "$operations"
exec 9>"$operations/study.lock"
flock -n 9
run() {
  python -m seis_ssl_cluster.hmm.single_head_distillation "$@"
}
run --phase validate
for phase in full embeddings downstream; do
  if [[ "$main_gpu" == "$volve_gpu" ]]; then
    for survey in f3 parihaka volve; do
      run --phase "$phase" --survey "$survey" --gpus "$main_gpu"
    done
  else
    (
      run --phase "$phase" --survey f3 --gpus "$main_gpu"
      run --phase "$phase" --survey parihaka --gpus "$main_gpu"
    ) &
    main_pid=$!
    run --phase "$phase" --survey volve --gpus "$volve_gpu" &
    volve_pid=$!
    status=0
    wait "$main_pid" || status=1
    wait "$volve_pid" || status=1
    if [[ "$status" != 0 ]]; then
      exit "$status"
    fi
  fi
done
run --phase summary
python - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
root = Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
summary = root / 'studies/hmm_v1/zero_distillation_ablation_v1/summary/summary.json'
receipt = root / 'operations/hmm_v1/zero_distillation_ablation_v1/analysis_complete.json'
with receipt.open('x') as stream:
    json.dump({'status': 'complete', 'utc': datetime.now(timezone.utc).isoformat(),
               'new_training_runs': 9, 'new_downstream_cells': 135,
               'reused_control_cells': 270, 'summary': str(summary),
               'summary_sha256': hashlib.sha256(summary.read_bytes()).hexdigest()}, stream, indent=2)
PY
