#!/usr/bin/env bash
set -euo pipefail

: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?set the artifact root}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
experiment=experiments/volve/horizon_benchmark_v1/33_missing_hmm_comparison_v1
suite=missing_hmm_comparison_v1
artifact_root=$SEIS_SSL_CLUSTER_ARTIFACT_ROOT

for init in rot90_asym_g060_3ep random_init; do
  clustering="$artifact_root/clustering/volve/horizon_benchmark_v1/$suite/$init/k6"
  targets="$artifact_root/pseudo_targets/volve/horizon_benchmark_v1/$suite/$init"
  if [[ ! -f "$targets/k6/volve_st10010.pseudo_target_metadata.json" ]]; then
    python proc/seis_ssl_cluster/cluster_embeddings.py --config "$experiment/10_hmm_targets/$init.yaml" --dry-run
    python proc/seis_ssl_cluster/cluster_embeddings.py --config "$experiment/10_hmm_targets/$init.yaml"
    python proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py \
      --clustering-output-dir "$clustering" --pseudo-target-root "$targets" \
      --k 6 --confidence 1.0 --boundary-alpha 0.0 --boundary-tau 1.0 --schema-version 2
  fi
done

for init in rot90_asym_g060_3ep random_init; do
  for distill in 010 020; do
    arm="${init}_hmm_k6_distill${distill}"
    config="$experiment/20_pretraining/$arm.yaml"
    output="$artifact_root/pretraining/volve/horizon_benchmark_v1/$suite/$arm/full_25ep"
    python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$config" --dry-run
    if [[ -f "$output/latest.pt" ]] && python proc/seis_ssl_cluster/audit_volve_horizon_recipe_arm_sources.py \
      --config "$experiment/40_downstream/$arm.yaml" --models "$arm"; then
      continue
    fi
    if [[ ! -f "$output/latest.pt" ]]; then
      python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$config" \
        --output-root "${output%/full_25ep}/smoke_2step" --max-steps 2
      python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$config"
    else
      python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$config" --resume "$output/latest.pt"
    fi
    python proc/seis_ssl_cluster/audit_volve_horizon_recipe_arm_sources.py \
      --config "$experiment/40_downstream/$arm.yaml" --models "$arm"
  done
done
