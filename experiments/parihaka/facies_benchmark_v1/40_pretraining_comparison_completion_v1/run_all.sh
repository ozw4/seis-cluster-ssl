#!/usr/bin/env bash
set -euo pipefail

: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?Set the shared artifact root}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1
experiment=experiments/parihaka/facies_benchmark_v1/40_pretraining_comparison_completion_v1
artifact_stem=pretraining_comparison_completion_v1
run_root="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/channel_benchmark/${artifact_stem}"
mkdir -p "$run_root/logs"
exec 9>"$run_root/driver.lock"
flock -n 9 || { echo 'Another Parihaka completion driver holds the lock'; exit 1; }

run_logged() {
  local name="$1"
  shift
  echo "$(date -u +%FT%TZ) START $name: $*"
  "$@" >"$run_root/logs/$name.log" 2>&1
  echo "$(date -u +%FT%TZ) DONE $name"
}

for source in local_bt3 random; do
  target_config="$experiment/10_hmm_targets/$source/01_cluster_hmm_k6.yaml"
  clustering_root="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/clustering/parihaka/facies_benchmark_v1/${artifact_stem}/$source/k6"
  pseudo_root="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pseudo_targets/parihaka/facies_benchmark_v1/${artifact_stem}/$source"
  if [[ ! -f "$pseudo_root/k6/parihaka.pseudo_target_metadata.json" ]]; then
    run_logged "${source}_cluster_dry_run" python proc/seis_ssl_cluster/cluster_embeddings.py --config "$target_config" --dry-run
    if [[ ! -f "$clustering_root/labels/k6/parihaka.cluster_label_metadata.json" ]]; then
      run_logged "${source}_cluster" python proc/seis_ssl_cluster/cluster_embeddings.py --config "$target_config"
    fi
    run_logged "${source}_export_dry_run" bash "$experiment/10_hmm_targets/$source/02_export_pseudo_targets.sh" --dry-run
    run_logged "${source}_export" bash "$experiment/10_hmm_targets/$source/02_export_pseudo_targets.sh"
  fi
done

for source in local_bt3 random; do
  if [[ "$source" == local_bt3 ]]; then
    model_stem=local_barlow_twins_rot90_asym_g060_3ep_hmm_k6
  else
    model_stem=random_init_self_target_hmm_k6
  fi
  for distill in 010 020; do
    model="${model_stem}_distill${distill}"
    train_config="$experiment/20_stage2/$source/distill$distill/01_full_25ep.yaml"
    train_root="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pretraining/parihaka/facies_benchmark_v1/${artifact_stem}/$source/distill$distill/full_25ep"
    checkpoint="$train_root/latest.pt"
    run_logged "${model}_train_dry_run" python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$train_config" --dry-run
    if ! python -m seis_ssl_cluster.parihaka.pretraining_comparison --checkpoint-complete "$checkpoint" --train-config "$train_config"; then
      resume_args=()
      if [[ -f "$checkpoint" ]]; then
        resume_args=(--resume "$checkpoint")
      else
        smoke_root="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pretraining/parihaka/facies_benchmark_v1/${artifact_stem}/$source/distill$distill/smoke_1step"
        if [[ ! -f "$smoke_root/latest.pt" ]]; then
          run_logged "${model}_smoke" python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$train_config" --max-steps 1 --output-root "$smoke_root"
        fi
      fi
      run_logged "${model}_train" python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$train_config" "${resume_args[@]}"
    fi
    python -m seis_ssl_cluster.parihaka.pretraining_comparison --checkpoint-complete "$checkpoint" --train-config "$train_config"
    embedding_config="$experiment/30_embeddings/01_extract_${model}.yaml"
    run_logged "${model}_embedding_dry_run" python proc/seis_ssl_cluster/extract_embeddings.py --config "$embedding_config" --device cuda --dry-run
    run_logged "${model}_embedding" python proc/seis_ssl_cluster/extract_embeddings.py --config "$embedding_config" --device cuda --skip-existing
    downstream_config="$experiment/40_downstream/01_${model}.yaml"
    for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do
      for size in small medium large; do
        job_dir="$run_root/runs/model=$model/layout=$layout/size=$size"
        if [[ -f "$job_dir/metrics.json" ]]; then
          continue
        fi
        resume_args=()
        if [[ -f "$job_dir/latest.pt" ]]; then resume_args=(--resume "$job_dir/latest.pt"); fi
        command=(python proc/seis_ssl_cluster/run_parihaka_channel_decoder.py --config "$downstream_config" --model "$model" --layout "$layout" --size "$size" --layout-config experiments/parihaka/facies_benchmark_v1/30_channel_benchmark_v1/02_layouts.yaml --device cuda)
        run_logged "${model}_${layout}_${size}_dry_run" "${command[@]}" --dry-run
        run_logged "${model}_${layout}_${size}" "${command[@]}" "${resume_args[@]}"
      done
    done
  done
done

model=mae_hmm_k6_distill010
downstream_config="$experiment/40_downstream/02_mae_hmm_k6_distill010.yaml"
for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do
  for size in small medium large; do
    job_dir="$run_root/runs/model=$model/layout=$layout/size=$size"
    if [[ -f "$job_dir/metrics.json" ]]; then continue; fi
    resume_args=()
    if [[ -f "$job_dir/latest.pt" ]]; then resume_args=(--resume "$job_dir/latest.pt"); fi
    command=(python proc/seis_ssl_cluster/run_parihaka_channel_decoder.py --config "$downstream_config" --model "$model" --layout "$layout" --size "$size" --layout-config experiments/parihaka/facies_benchmark_v1/30_channel_benchmark_v1/02_layouts.yaml --device cuda)
    if [[ "$size" == medium ]]; then
      validation_source="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/channel_benchmark/hmm_distillation_weight_v1/validation_runs/model=$model/layout=$layout/size=$size"
      command+=(--evaluate-completed-validation "$validation_source")
    fi
    run_logged "${model}_${layout}_${size}_dry_run" "${command[@]}" --dry-run
    run_logged "${model}_${layout}_${size}" "${command[@]}" "${resume_args[@]}"
  done
done
run_logged summary python -m seis_ssl_cluster.parihaka.pretraining_comparison --summary-config "$experiment/50_summary/01_all_nine_arms.yaml"
echo "$(date -u +%FT%TZ) COMPLETE: Parihaka nine-arm comparison"
