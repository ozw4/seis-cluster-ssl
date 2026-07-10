#!/usr/bin/env bash
set -euo pipefail

python proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py \
  --clustering-output-dir /workspace/artifacts/seis_ssl_cluster/clustering/f3/facies_benchmark_v1/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/full/overlap_x16/strat_hmm_k6_pca64_resid_token_phase_edge8_expected3_iter10 \
  --pseudo-target-root /workspace/artifacts/seis_ssl_cluster/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_k6_pca64_resid_token_phase_edge8_expected3_iter10_bootstrap_boundary_a000_t2_parity \
  --k 6 \
  --confidence 1.0 \
  --boundary-alpha 0.0 \
  --boundary-tau 2.0 \
  "$@"
