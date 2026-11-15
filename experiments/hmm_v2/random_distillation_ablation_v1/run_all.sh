#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export SEIS_SSL_CLUSTER_WORKSPACE="$PWD"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?Set the existing artifact root}"
: "${F3_ROOT:?Use the frozen F3 label path}"
: "${SEIS_SSL_CLUSTER_VOLVE_ROOT:?Set Volve data root}"
read -r -a devices <<< "${SENSITIVITY_GPUS:-0 1}"
matrix=experiments/hmm_v2/random_distillation_ablation_v1/matrix.yaml
operations="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/operations/hmm_v2/random_distillation_ablation_v1"
mkdir -p "$operations"
exec 9>"$operations/study.lock"
flock -n 9
surveys=(f3 parihaka volve)
run() {
  python -m seis_ssl_cluster.hmm.distillation_sensitivity --matrix "$matrix" "$@"
}
run_phase() {
  local phase="$1" index pid status=0
  local pids=()
  for index in "${!surveys[@]}"; do
    CUDA_VISIBLE_DEVICES="${devices[index % ${#devices[@]}]}" \
      run --phase "$phase" --survey "${surveys[index]}" --weight 0.0 --resume &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  return "$status"
}
run --phase validate
for phase in full embeddings; do
  run_phase "$phase"
done
# This evaluates retained baseline weights; it never trains a 0.2 model.
CUDA_VISIBLE_DEVICES="${devices[0]}" python -m seis_ssl_cluster.hmm.distillation_inputs \
  --matrix "$matrix" --diagnose-baselines
for phase in medium remaining; do
  run_phase "$phase"
done
python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.hmm.distillation_sensitivity import LAYOUTS, SIZES, _cell_path
matrix = load_config('experiments/hmm_v2/random_distillation_ablation_v1/matrix.yaml')
cells = sum(_cell_path(s, arms['0.0'], l, z).is_file()
            for s, arms in matrix['surveys'].items() for l in LAYOUTS for z in SIZES)
if cells != 45:
    raise RuntimeError(f'Random decoder matrix incomplete: {cells}/45')
output = Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT']) / 'operations/hmm_v2/random_distillation_ablation_v1/experiments_complete.json'
if not output.exists():
    with output.open('x') as stream:
        json.dump({'status': 'experiments_complete', 'utc': datetime.now(timezone.utc).isoformat(),
                   'new_full_training_completed': 3, 'new_embeddings_completed': 3,
                   'new_downstream_cells': cells, 'baseline_models_reused': 3,
                   'baseline_cells_reused': 45, 'additional_experiment_audits': False}, stream, indent=2)
        stream.write('\n')
PY
run --phase test --resume
run --phase summary
python -m seis_ssl_cluster.hmm.distillation_review --random
