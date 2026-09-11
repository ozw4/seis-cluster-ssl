#!/usr/bin/env bash
set -euo pipefail
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?Set the shared artifact root}"
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export staging_experiment=experiments/parihaka/facies_benchmark_v1/40_pretraining_comparison_completion_v1
export staging_config="$staging_experiment/40_downstream/03_mae_hmm_k6_distill010_gpu1_staging.yaml"
export staging_root="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/channel_benchmark/pretraining_comparison_completion_v1/gpu1_staging"
export staging_model=mae_hmm_k6_distill010
mkdir -p "$staging_root/logs"
exec 9>"$staging_root/driver.lock"
flock -n 9 || { echo 'GPU 1 staging driver already running'; exit 1; }

audit_cell() {
  python - "$1" <<'PY'
import json
import sys
from pathlib import Path
from seis_ssl_cluster.parihaka.channel_completion import inspect_completed_channel_job
print(json.dumps(inspect_completed_channel_job(Path(sys.argv[1]))))
PY
}
export -f audit_cell

run_cell() {
  set -euo pipefail
  local layout="$1" size="$2"
  local job_dir="$staging_root/model=$staging_model/layout=$layout/size=$size"
  local log_stem="$staging_root/logs/${layout}_${size}"
  if [[ -f "$job_dir/metrics.json" ]]; then
    audit_cell "$job_dir" > "${log_stem}_audit.log" 2>&1
    echo "$(date -u +%FT%TZ) SKIP COMPLETE $layout/$size"
    return
  fi
  local resume_args=()
  if [[ -f "$job_dir/latest.pt" ]]; then resume_args=(--resume "$job_dir/latest.pt"); fi
  local command=(python proc/seis_ssl_cluster/run_parihaka_channel_decoder.py --config "$staging_config" --model "$staging_model" --layout "$layout" --size "$size" --layout-config experiments/parihaka/facies_benchmark_v1/30_channel_benchmark_v1/02_layouts.yaml --device cuda)
  echo "$(date -u +%FT%TZ) DRY RUN $layout/$size"
  "${command[@]}" --dry-run > "${log_stem}_dry_run.log" 2>&1
  echo "$(date -u +%FT%TZ) TRAIN $layout/$size (GPU 1)"
  "${command[@]}" "${resume_args[@]}" > "${log_stem}_train.log" 2>&1
  audit_cell "$job_dir" > "${log_stem}_audit.log" 2>&1
  echo "$(date -u +%FT%TZ) COMPLETE $layout/$size"
}
export -f run_cell

for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do
  for size in small large; do
    printf '%s %s\n' "$layout" "$size"
  done
done | xargs -n 2 -P 3 bash -c 'run_cell "$1" "$2"' _
echo "$(date -u +%FT%TZ) COMPLETE: all ten GPU 1 staging cells; canonical publication remains separate"
