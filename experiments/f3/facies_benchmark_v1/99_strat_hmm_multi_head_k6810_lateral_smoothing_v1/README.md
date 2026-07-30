# F3 M5-LS lateral hard-target pretraining

M5-LS changes only the offline hard pseudo-target labels from the selected
`mh_nocons` baseline. It uses one XY four-neighbour, source-embedding cosine-RBF
lateral mean-field message and reprojects once with the original ordered
Viterbi path. Training consumes only hard labels through the existing hard
multi-head dataset, collate, and loss route; posterior tensors never enter a
training batch.

The candidate set and target-only selection rule were fixed before diagnostics:
`beta010 = 0.10`, `beta025 = 0.25`, and `beta050 = 0.50`. The selected beta is
the numerically smallest eligible candidate under
`target_only_smallest_eligible_beta_v1`. No facies or lithology labels,
decoder outputs, or downstream metrics are read by this workflow.

```bash
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/path/to/artifacts/seis_ssl_cluster
export EXP=experiments/f3/facies_benchmark_v1/99_strat_hmm_multi_head_k6810_lateral_smoothing_v1

python proc/seis_ssl_cluster/export_strat_hmm_multi_head_lateral_targets.py \
  --config "$EXP/01_export_lateral_beta010.yaml" --dry-run
python proc/seis_ssl_cluster/export_strat_hmm_multi_head_lateral_targets.py \
  --config "$EXP/01_export_lateral_beta010.yaml"
python proc/seis_ssl_cluster/export_strat_hmm_multi_head_lateral_targets.py \
  --config "$EXP/01_export_lateral_beta010.yaml" --only-missing

python proc/seis_ssl_cluster/export_strat_hmm_multi_head_lateral_targets.py \
  --config "$EXP/02_export_lateral_beta025.yaml" --dry-run
python proc/seis_ssl_cluster/export_strat_hmm_multi_head_lateral_targets.py \
  --config "$EXP/02_export_lateral_beta025.yaml"
python proc/seis_ssl_cluster/export_strat_hmm_multi_head_lateral_targets.py \
  --config "$EXP/02_export_lateral_beta025.yaml" --only-missing

python proc/seis_ssl_cluster/export_strat_hmm_multi_head_lateral_targets.py \
  --config "$EXP/03_export_lateral_beta050.yaml" --dry-run
python proc/seis_ssl_cluster/export_strat_hmm_multi_head_lateral_targets.py \
  --config "$EXP/03_export_lateral_beta050.yaml"
python proc/seis_ssl_cluster/export_strat_hmm_multi_head_lateral_targets.py \
  --config "$EXP/03_export_lateral_beta050.yaml" --only-missing

python proc/seis_ssl_cluster/calibrate_f3_m5_lateral_targets.py \
  --config "$EXP/04_calibrate_lateral_targets.yaml" --dry-run
python proc/seis_ssl_cluster/calibrate_f3_m5_lateral_targets.py \
  --config "$EXP/04_calibrate_lateral_targets.yaml"
python proc/seis_ssl_cluster/calibrate_f3_m5_lateral_targets.py \
  --config "$EXP/04_calibrate_lateral_targets.yaml" --only-missing

python proc/seis_ssl_cluster/validate_f3_m5_lateral_smoothing_pretraining.py \
  --config "$EXP/08_validate_lateral_pretraining.yaml" --phase targets
```

Publish the lightweight target-only review files after calibration. This reads
only the calibration handoff and report; it never copies target arrays.

```bash
export CALIBRATION_DIR="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_lateral_mean_field_selected_v1"
export RESULTS_DIR="results/f3/facies_benchmark_v1/strat_hmm_multi_head_k6810_lateral_smoothing_v1"

python proc/seis_ssl_cluster/publish_f3_m5_lateral_smoothing_results.py \
  --artifact-root "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT" \
  --workspace-root "$PWD" \
  --calibration-handoff "$CALIBRATION_DIR/lateral_target_calibration_handoff.json" \
  --calibration-report "$CALIBRATION_DIR/lateral_target_calibration_report.json" \
  --output-dir "$RESULTS_DIR"
```

Calibration includes a read-only beta-zero parity replay. It must be bitwise
identical to the frozen source hard labels and valid masks for K=6/8/10. A
candidate must satisfy every target-only eligibility diagnostic. If the result
is `M5_LS_TARGET_HOLD`, do not run smoke or full pretraining. The target-only
calibration report is still the complete publication for that result.

When calibration reports `M5_LS_TARGET_SELECTED`, its selected manifest is a
byte-exact immutable copy of the selected candidate manifest. It references
the selected candidate arrays and contains no copied arrays. The smoke root is
isolated from the full root:

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/05_train_lateral_smoke.yaml" --dry-run --device cpu --max-steps 2
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/05_train_lateral_smoke.yaml" --device cpu --max-steps 2
python proc/seis_ssl_cluster/validate_f3_m5_lateral_smoothing_pretraining.py \
  --config "$EXP/08_validate_lateral_pretraining.yaml" --phase targets
python proc/seis_ssl_cluster/validate_f3_m5_lateral_smoothing_pretraining.py \
  --config "$EXP/08_validate_lateral_pretraining.yaml" --phase smoke
```

Full training, embedding extraction, and original-split screening are later
issues and are not run here. Their eventual commands are:

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/06_train_lateral_full.yaml" --dry-run
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/06_train_lateral_full.yaml"
python proc/seis_ssl_cluster/validate_f3_m5_lateral_smoothing_pretraining.py \
  --config "$EXP/08_validate_lateral_pretraining.yaml" --phase checkpoints
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/07_extract_lateral_embeddings.yaml" --skip-existing
python proc/seis_ssl_cluster/validate_f3_m5_lateral_smoothing_pretraining.py \
  --config "$EXP/08_validate_lateral_pretraining.yaml" --phase complete
```

Hard M4, M5-U posterior, and M5-LS lateral checkpoints have distinct
representation identities and cannot cross-resume.
