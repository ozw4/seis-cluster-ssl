#!/usr/bin/env bash
set -euo pipefail

dry_run=false
if [[ $# -gt 1 ]]; then
  echo 'Usage: run_hmm_lane.sh [--dry-run]' >&2
  exit 2
fi
case "${1:-}" in
  '') ;;
  --dry-run) dry_run=true ;;
  *) echo 'Usage: run_hmm_lane.sh [--dry-run]' >&2; exit 2 ;;
esac

: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?Set the shared artifact root}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1
experiment=experiments/parihaka/facies_benchmark_v1/40_pretraining_comparison_completion_v1
run_root="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/channel_benchmark/pretraining_comparison_completion_v1"
pretraining_root="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pretraining/parihaka/facies_benchmark_v1/pretraining_comparison_completion_v1"

# Refuse to launch using a checkout without the shared future-CLI coordination.
python -c 'from seis_ssl_cluster.parihaka.coordinated_hmm import run_coordinated_hmm_if_scoped; assert callable(run_coordinated_hmm_if_scoped)'

verify_driver_lock() {
  python -c 'import sys; from pathlib import Path; from seis_ssl_cluster.parihaka.coordinated_hmm import verify_open_lock; verify_open_lock(Path(sys.argv[1]), 9)' "$run_root/hmm_lane_driver.lock"
}

if [[ "$dry_run" == false ]]; then
  mkdir -p "$run_root/logs"
  if [[ -L "$run_root/hmm_lane_driver.lock" ]] || { [[ -e "$run_root/hmm_lane_driver.lock" ]] && [[ ! -f "$run_root/hmm_lane_driver.lock" ]]; }; then
    echo 'Refusing a symlinked or non-regular HMM lane driver lock' >&2
    exit 1
  fi
  exec 9<>"$run_root/hmm_lane_driver.lock"
  verify_driver_lock
  flock -n 9 || { echo 'Another auxiliary HMM lane driver is running' >&2; exit 1; }
  verify_driver_lock
  lane_logs=$(mktemp -d "$run_root/logs/hmm_lane_XXXXXXXX")
  echo "$(date -u +%FT%TZ) LOG DIRECTORY $lane_logs"
fi

run_logged() {
  local name="$1"
  shift
  echo "$(date -u +%FT%TZ) START $name: $*"
  if [[ "$dry_run" == true ]]; then
    "$@"
  else
    "$@" >"$lane_logs/$name.log" 2>&1
  fi
  echo "$(date -u +%FT%TZ) DONE $name"
}

for arm in local_bt3/020 random/010 random/020; do
  source="${arm%/*}"
  distill="${arm#*/}"
  train_config="$experiment/20_stage2/$source/distill$distill/01_full_25ep.yaml"
  source_root="$pretraining_root/$source/distill$distill"
  command=(python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$train_config")
  run_logged "${source}_${distill}_full_dry_run" "${command[@]}" --dry-run
  if [[ "$dry_run" == true ]]; then
    run_logged "${source}_${distill}_smoke_dry_run" "${command[@]}" --max-steps 1 --output-root "$source_root/smoke_1step" --dry-run
    continue
  fi
  # Both calls share the same lane and per-arm locks with future main children.
  # The helper rechecks completion and partial-run ownership after acquiring them.
  run_logged "${source}_${distill}_smoke" "${command[@]}" --max-steps 1 --output-root "$source_root/smoke_1step"
  run_logged "${source}_${distill}_full" "${command[@]}"
  run_logged "${source}_${distill}_budget_audit" python -m seis_ssl_cluster.parihaka.pretraining_comparison --checkpoint-complete "$source_root/full_25ep/latest.pt" --train-config "$train_config"
done

if [[ "$dry_run" == true ]]; then
  echo 'DRY RUN COMPLETE: no locks, logs, or training outputs created'
else
  echo "$(date -u +%FT%TZ) COMPLETE: three remaining Parihaka HMM sources; main driver owns embedding and downstream"
fi
