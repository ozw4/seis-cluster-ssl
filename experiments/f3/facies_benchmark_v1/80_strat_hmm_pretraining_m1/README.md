# F3 Strat HMM Pretraining Milestone 1

This experiment wires a reproducible F3 milestone where stratigraphic HMM labels
are used as a structured pretext task. The HMM labels are pseudo-targets for
student pretraining, not final lithology labels and not final evaluation output.

## Inputs

- MAE teacher checkpoint:
  `/workspace/artifacts/seis_ssl_cluster/pretraining/nopims/pretrain_v1/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/full_100ep/mae_best.pt`
- F3 amplitude manifest:
  `/workspace/artifacts/seis_ssl_cluster/registry/manifests/f3/facies_benchmark_v1/f3_amplitude_manifest.json`
- Existing F3 embeddings:
  `/workspace/artifacts/seis_ssl_cluster/embeddings/f3/facies_benchmark_v1/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/overlap_x16`
- Existing HMM clustering output, defaulting to the k6 expected-boundary path-prior run:
  `/workspace/artifacts/seis_ssl_cluster/clustering/f3/facies_benchmark_v1/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/full/overlap_x16/strat_hmm_k6_pca64_resid_token_phase_edge8_expected3_iter10`
- F3 lithology token-label inputs from the existing `50_lithology` workflow.

Edit paths when your artifact root or data root differs from the repository
defaults.

## Smoke Run

Export bootstrap HMM labels to pseudo-target artifacts:

```bash
bash experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1/01_export_bootstrap_hmm_pseudo_targets.sh
```

Dry-run, then run the CPU smoke training config:

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1/02_train_single_head_topblock_distill_smoke.yaml \
  --dry-run \
  --device cpu \
  --max-steps 2

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1/02_train_single_head_topblock_distill_smoke.yaml \
  --device cpu \
  --max-steps 2
```

Refresh pseudo-targets from smoke prototype logits only as a wiring check:

```bash
python proc/seis_ssl_cluster/build_strat_hmm_pseudo_targets.py \
  --config experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1/08_refresh_pseudo_targets_from_logits_smoke.yaml \
  --dry-run \
  --device cpu \
  --overwrite
```

## Full Run

Train the single-head, k6, top-block distillation student:

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1/03_train_single_head_topblock_distill_full.yaml \
  --dry-run

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1/03_train_single_head_topblock_distill_full.yaml
```

Extract student embeddings with the existing extraction command:

```bash
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1/04_extract_student_embeddings.yaml
```

Build the token dataset, train the same few-label lithology probe, and build the
report:

```bash
python proc/seis_ssl_cluster/build_f3_lithology_token_dataset.py \
  --config experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1/05_build_lithology_token_dataset.yaml

python proc/seis_ssl_cluster/train_f3_lithology_probe.py \
  --config experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1/06_train_lithology_probe.yaml

python proc/seis_ssl_cluster/build_f3_lithology_report.py \
  --config experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1/07_build_lithology_report.yaml

python proc/seis_ssl_cluster/build_f3_lithology_comparison_report.py \
  --config experiments/f3/facies_benchmark_v1/50_lithology_baselines/05_build_baseline_comparison_report.yaml
```

## Expected Outputs

- Bootstrap pseudo-targets under
  `/workspace/artifacts/seis_ssl_cluster/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_k6_pca64_resid_token_phase_edge8_expected3_iter10_bootstrap`.
- Student checkpoints under
  `/workspace/artifacts/seis_ssl_cluster/pretraining/f3/facies_benchmark_v1/strat_hmm_pretext_m1_k6_topblock1_distill`.
- Extraction-compatible checkpoints: use `best.pt` or `latest.pt` with
  `proc/seis_ssl_cluster/extract_embeddings.py`; the extraction path reads the
  student encoder and ignores the ordered prototype head.
- Student embeddings under
  `/workspace/artifacts/seis_ssl_cluster/embeddings/f3/facies_benchmark_v1/strat_hmm_pretext_m1_k6_topblock1_distill/overlap_x16`.
- Lithology probe reports under
  `/workspace/artifacts/seis_ssl_cluster/lithology/f3/facies_benchmark_v1/strat_hmm_pretext_m1_k6_topblock1_distill/overlap_x16/png_slices_segy_labels_v1`.

## Evaluation

Compare few-label macro F1 and mean IoU against the existing MAE baseline,
random encoder, z-only, and xyz-coordinate baselines from `50_lithology` and
`50_lithology_baselines`. The HMM map itself is not the final task output.

Guardrails to run or report before expanding the method:

- z-only HMM target
- shuffled HMM target
- no-HMM prototype-only training
- distillation-only training
- MAE continuation or top-block-only baseline, when available

If HMM maps look cleaner but the downstream few-label probe does not improve, do
not proceed to lateral smoothing or multi-resolution heads.
