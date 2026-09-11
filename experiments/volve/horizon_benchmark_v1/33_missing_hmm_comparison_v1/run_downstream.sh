#!/usr/bin/env bash
set -euo pipefail

: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?set the artifact root}"
: "${SEIS_SSL_CLUSTER_VOLVE_ROOT:?set the public Volve root}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
experiment=experiments/volve/horizon_benchmark_v1/33_missing_hmm_comparison_v1
layouts=experiments/volve/horizon_benchmark_v1/20_horizon_supervision/01_layouts.yaml
runs="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/horizon/volve/horizon_benchmark_v1/local_bt_recipe_arm_v1/runs"

for init in rot90_asym_g060_3ep random_init; do
  for distill in 010 020; do
    arm="${init}_hmm_k6_distill${distill}"
    python proc/seis_ssl_cluster/audit_volve_horizon_recipe_arm_sources.py \
      --config "$experiment/40_downstream/$arm.yaml" --models "$arm"
    python proc/seis_ssl_cluster/extract_embeddings.py \
      --config "$experiment/30_embeddings/$arm.yaml" --device cuda --dry-run
    python proc/seis_ssl_cluster/extract_embeddings.py \
      --config "$experiment/30_embeddings/$arm.yaml" --device cuda --skip-existing
    config="$experiment/40_downstream/$arm.yaml"
    for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do
      for size in small medium large; do
        cell="$runs/model=$arm/layout=$layout/size=$size"
        python proc/seis_ssl_cluster/run_volve_horizon_recipe_arm.py \
          --config "$config" --model "$arm" --layout "$layout" --size "$size" \
          --layout-config "$layouts" --dry-run
        if [[ -f "$cell/metrics.json" ]]; then
          continue
        fi
        resume=()
        if [[ -f "$cell/latest.pt" ]]; then
          resume=(--resume "$cell/latest.pt")
        fi
        python proc/seis_ssl_cluster/run_volve_horizon_recipe_arm.py \
          --config "$config" --model "$arm" --layout "$layout" --size "$size" \
          --layout-config "$layouts" --device cuda "${resume[@]}"
      done
    done
    python proc/seis_ssl_cluster/summarize_volve_horizon_recipe_arm.py --config "$config" --check-only
    python proc/seis_ssl_cluster/summarize_volve_horizon_recipe_arm.py --config "$config"
  done
done
