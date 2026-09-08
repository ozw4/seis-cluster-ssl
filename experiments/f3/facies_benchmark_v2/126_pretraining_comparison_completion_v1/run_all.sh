#!/usr/bin/env bash
# Resume-safe completion of the three missing F3 arms, one GPU process at a time.
set -euo pipefail
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?export artifact root first}"
: "${SEIS_SSL_CLUSTER_WORKSPACE:?export canonical repository root first}"
: "${F3_ROOT:?export F3 data root first}"
: "${CUDA_VISIBLE_DEVICES:?choose an available GPU before running}"

experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_dir="$(cd "$experiment_dir/../../../.." && pwd)"
cd "$repository_dir"
export PYTHONPATH="$repository_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
log_root="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/logs/f3/pretraining_comparison_completion_v1"
mkdir -p "$log_root"
exec 9>"$log_root/driver.lock"
flock -n 9
trap 'result=$?; printf "%s exit=%s\n" "$(date -u +%FT%TZ)" "$result" >> "$log_root/driver_events.log"' EXIT

run_stage() {
	local name="$1"
	shift
	printf '%s start %s\n' "$(date -u +%FT%TZ)" "$name" | tee -a "$log_root/driver_events.log"
	"$@" > "$log_root/$name.log" 2>&1
	printf '%s complete %s\n' "$(date -u +%FT%TZ)" "$name" | tee -a "$log_root/driver_events.log"
}

checkpoint_state() {
	python - "$1" <<'PY'
import sys
from pathlib import Path

import torch

from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config

path = Path(sys.argv[1])
expected = resolve_strat_hmm_pretext_config(load_config(path))
checkpoint_path = Path(expected['paths']['output_root']) / 'latest.pt'
if not checkpoint_path.exists():
	print('missing')
	sys.exit(0)
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
actual = checkpoint['stratigraphy_config']
for section in ('identity', 'teacher', 'student', 'pseudo_targets', 'loss', 'model', 'head'):
	if actual[section] != expected[section]:
		raise ValueError(f'{checkpoint_path}: {section} identity mismatch')
for key in ('epochs', 'samples_per_epoch', 'batch_size', 'seed', 'amp', 'lr', 'encoder_lr', 'max_steps'):
	if actual['train'][key] != expected['train'][key]:
		raise ValueError(f'{checkpoint_path}: train.{key} mismatch')
if actual['paths']['output_root'] != expected['paths']['output_root']:
	raise ValueError(f'{checkpoint_path}: output root mismatch')
train = expected['train']
steps = train['epochs'] * (train['samples_per_epoch'] // train['batch_size'])
if train['max_steps'] is not None:
	steps = min(steps, train['max_steps'])
epoch = int(checkpoint['epoch'])
step = int(checkpoint['global_step'])
if epoch > train['epochs'] or step > steps:
	raise ValueError(f'{checkpoint_path}: checkpoint exceeds requested budget')
print('complete' if epoch == train['epochs'] and step == steps else 'incomplete')
PY
}

arms=(
	mae100_hmm_k6_distill010
	local_bt_rot90_asym_g060_3ep_hmm_k6_distill010
	random_self_target_hmm_k6_distill020
)
for arm in "${arms[@]}"; do
	for stage in 01_smoke 02_full_25ep; do
		config="$experiment_dir/10_pretraining/$arm/$stage.yaml"
		state="$(checkpoint_state "$config")"
		if [[ "$state" == complete ]]; then
			printf '%s validated complete %s %s\n' "$(date -u +%FT%TZ)" "$arm" "$stage" | tee -a "$log_root/driver_events.log"
			continue
		fi
		run_stage "${arm}_${stage}_dry_run" python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$config" --dry-run
		resume_args=()
		if [[ "$state" == incomplete ]]; then
			subdir=full_25ep
			[[ "$stage" == 01_smoke ]] && subdir=gpu_feasibility_1step
			resume_args=(--resume "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/f3/facies_benchmark_v1/pretraining_comparison_completion_v1/$arm/$subdir/latest.pt")
		fi
		run_stage "${arm}_${stage}" python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$config" "${resume_args[@]}"
		[[ "$(checkpoint_state "$config")" == complete ]]
	done
done

for arm in "${arms[@]}"; do
	embedding_config="$experiment_dir/20_embeddings/$arm.yaml"
	run_stage "${arm}_embeddings_dry_run" python proc/seis_ssl_cluster/extract_embeddings.py --config "$embedding_config" --dry-run
	run_stage "${arm}_embeddings" python proc/seis_ssl_cluster/extract_embeddings.py --config "$embedding_config" --skip-existing
	candidate_config="$experiment_dir/30_downstream/$arm.yaml"
	run_stage "${arm}_candidate_dry_run" python proc/seis_ssl_cluster/run_f3_lithology_candidate.py --config "$candidate_config" --layout layout_000 --size small --dry-run
	for size in small medium large; do
		for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do
			cell_root="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/f3_lithology_benchmark/pretraining_comparison_completion_v1/runs/model=$arm/layout=$layout/size=$size"
			if [[ -f "$cell_root/evaluation/metrics.json" ]]; then
				continue
			fi
			resume_args=()
			if [[ -f "$cell_root/decoder/latest.pt" ]]; then
				resume_args=(--resume "$cell_root/decoder/latest.pt")
			fi
			run_stage "${arm}_${size}_${layout}" python proc/seis_ssl_cluster/run_f3_lithology_candidate.py --config "$candidate_config" --layout "$layout" --size "$size" "${resume_args[@]}"
		done
	done
	# Both producers audit all 15 completed cell identities before reporting.
	for summary in summarize_f3_lithology_candidate summarize_f3_lithology_candidate_five_way; do
		summary_subdir=summary
		[[ "$summary" == summarize_f3_lithology_candidate_five_way ]] && summary_subdir=summary_five_way
		if [[ "$summary" == summarize_f3_lithology_candidate_five_way ]]; then
			run_stage "${arm}_${summary}_dry_run" python "proc/seis_ssl_cluster/$summary.py" --config "$candidate_config" --dry-run
		fi
		if [[ ! -f "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/f3_lithology_benchmark/pretraining_comparison_completion_v1/$summary_subdir/$arm/summary.json" ]]; then
			run_stage "${arm}_${summary}" python "proc/seis_ssl_cluster/$summary.py" --config "$candidate_config"
		fi
	done
done
printf '%s all_complete 3 arms 45 cells\n' "$(date -u +%FT%TZ)" | tee -a "$log_root/driver_events.log"
