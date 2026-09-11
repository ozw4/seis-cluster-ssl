#!/usr/bin/env bash
set -euo pipefail
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?Set the shared artifact root}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1
experiment=experiments/parihaka/facies_benchmark_v1/40_pretraining_comparison_completion_v1
run_root="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/channel_benchmark/pretraining_comparison_completion_v1"
model=mae_hmm_k6_distill010
mkdir -p "$run_root/logs"
exec 9>"$run_root/medium_test.lock"
flock -n 9 || { echo 'Medium test evaluator already running'; exit 1; }
for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do
  job_dir="$run_root/runs/model=$model/layout=$layout/size=medium"
  if [[ -f "$job_dir/metrics.json" ]]; then continue; fi
  validation_source="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/channel_benchmark/hmm_distillation_weight_v1/validation_runs/model=$model/layout=$layout/size=medium"
  command=(python proc/seis_ssl_cluster/run_parihaka_channel_decoder.py --config "$experiment/40_downstream/02_mae_hmm_k6_distill010.yaml" --model "$model" --layout "$layout" --size medium --layout-config experiments/parihaka/facies_benchmark_v1/30_channel_benchmark_v1/02_layouts.yaml --device cuda --evaluate-completed-validation "$validation_source")
  echo "$(date -u +%FT%TZ) START $layout dry-run"
  "${command[@]}" --dry-run > "$run_root/logs/${model}_${layout}_medium_test_dry_run.log" 2>&1
  echo "$(date -u +%FT%TZ) START $layout test evaluation"
  "${command[@]}" > "$run_root/logs/${model}_${layout}_medium_test.log" 2>&1
  echo "$(date -u +%FT%TZ) DONE $layout test evaluation"
done
