# Current-code single-head K=6 control

This experiment retrains the historical M1 single-head K=6 pretext condition
with the current code, under the new identity
`strat_hmm_pretext_m1_current_k6_topblock1_distill_v1`. It is the primary
baseline for the forthcoming multi-head K=6/8/10 comparisons. Historical M1,
MAE, M2-A, their embeddings, and their decoder jobs are read-only inputs.

The scientific claim is limited to the F3 original split, cap25/cap50/cap100,
the fixed frozen voxel decoder, and five paired subsample seeds. This stage
does not implement a multi-head model, rerun historical decoder jobs, or run a
six-split analysis.

## Execution order

Run from `/workspace`:

```bash
export ROOT=/workspace/artifacts/seis_ssl_cluster
export EXP_CONTROL=experiments/f3/facies_benchmark_v1/93_strat_hmm_m1_current_k6_control
export CONTROL_PRETRAIN="$ROOT/pretraining/f3/facies_benchmark_v1/strat_hmm_pretext_m1_current_k6_topblock1_distill_v1"
export CONTROL_EMBEDDINGS="$ROOT/embeddings/f3/facies_benchmark_v1/strat_hmm_pretext_m1_current_k6_topblock1_distill_v1/overlap_x16"
export CONTROL_VOXEL="$ROOT/lithology/f3/facies_benchmark_v1/voxel_label_budget_current_k6_control_v1/original_split"
export CONTROL_REPORTS="$CONTROL_VOXEL/reports"
```

First validate the migration decision and immutable input identities. This
writes the control-only preflight manifest; it does not modify historical M1.

```bash
python proc/seis_ssl_cluster/validate_f3_current_k6_control.py preflight \
  --config "$EXP_CONTROL/02_train_current_k6_full.yaml"
```

Run the CPU two-step smoke. Its sibling output root is intentionally not a
resume source for the full run.

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP_CONTROL/01_train_current_k6_smoke.yaml" \
  --dry-run --device cpu --max-steps 2

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP_CONTROL/01_train_current_k6_smoke.yaml" \
  --device cpu --max-steps 2
```

Then validate and run the full 25-epoch pretraining condition.

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP_CONTROL/02_train_current_k6_full.yaml" --dry-run

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP_CONTROL/02_train_current_k6_full.yaml"
```

Only an incomplete full run may be resumed, and only from its own `latest.pt`:

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP_CONTROL/02_train_current_k6_full.yaml" \
  --resume "$CONTROL_PRETRAIN/latest.pt"
```

After completion (`epoch=25`, `global_step=25600`), validate the checkpoint,
freeze contract, parameter groups, provenance, and best-checkpoint criterion.

```bash
python proc/seis_ssl_cluster/validate_f3_current_k6_control.py checkpoint \
  --config "$EXP_CONTROL/02_train_current_k6_full.yaml" \
  --reports-dir "$CONTROL_REPORTS"
```

Extract frozen embeddings with the canonical memmap cache path, then validate
their shape, dtype, finite values, current-control checkpoint binding, and
historical valid-token-mask identity.

```bash
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP_CONTROL/03_extract_current_k6_embeddings.yaml" --dry-run

python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP_CONTROL/03_extract_current_k6_embeddings.yaml" --skip-existing

python proc/seis_ssl_cluster/validate_f3_current_k6_control.py embeddings \
  --embeddings-dir "$CONTROL_EMBEDDINGS" \
  --checkpoint "$CONTROL_PRETRAIN/best.pt" \
  --reports-dir "$CONTROL_REPORTS"
```

The full-label token sanity check reuses the historical M1 token-row split and
support exactly. Run each long stage only after its dry-run resolves.

```bash
python proc/seis_ssl_cluster/build_f3_lithology_token_dataset.py \
  --config "$EXP_CONTROL/04_build_current_k6_token_dataset.yaml" --dry-run
python proc/seis_ssl_cluster/build_f3_lithology_token_dataset.py \
  --config "$EXP_CONTROL/04_build_current_k6_token_dataset.yaml"

python proc/seis_ssl_cluster/train_f3_lithology_probe.py \
  --config "$EXP_CONTROL/05_train_current_k6_token_probe.yaml" --dry-run
python proc/seis_ssl_cluster/train_f3_lithology_probe.py \
  --config "$EXP_CONTROL/05_train_current_k6_token_probe.yaml"

python proc/seis_ssl_cluster/build_f3_lithology_report.py \
  --config "$EXP_CONTROL/06_build_current_k6_token_report.yaml" --dry-run
python proc/seis_ssl_cluster/build_f3_lithology_report.py \
  --config "$EXP_CONTROL/06_build_current_k6_token_report.yaml"

python proc/seis_ssl_cluster/validate_f3_current_k6_control.py token-probe \
  --current-metrics "$ROOT/lithology/f3/facies_benchmark_v1/strat_hmm_pretext_m1_current_k6_topblock1_distill_v1/overlap_x16/png_slices_segy_labels_v1/probes/linear_balanced_v1/metrics.json" \
  --output "$CONTROL_REPORTS/token_probe_comparison.csv"
```

The control runner validates the existing MAE/M1 M3-V-LB manifest and adds
only the 15 current-K6 jobs. `--only-missing` revalidates completed jobs,
resumes valid incomplete `latest.pt` checkpoints, and quarantines mismatches;
`--resume` is restricted to selected incomplete jobs with valid `latest.pt`.

```bash
python proc/seis_ssl_cluster/run_f3_lithology_voxel_label_budget_control.py \
  --config "$EXP_CONTROL/07_run_current_k6_voxel_label_budget.yaml" \
  --dry-run --device auto

python proc/seis_ssl_cluster/run_f3_lithology_voxel_label_budget_control.py \
  --config "$EXP_CONTROL/07_run_current_k6_voxel_label_budget.yaml" \
  --only-missing --device auto

python proc/seis_ssl_cluster/run_f3_lithology_voxel_label_budget_control.py \
  --config "$EXP_CONTROL/07_run_current_k6_voxel_label_budget.yaml" \
  --only-missing --device auto
```

Finally, the summary revalidates all paired identities, reaggregates the
historical M1-minus-MAE rows against the published M3-V-LB values, writes the
handoff, and publishes only lightweight report files.

```bash
python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_label_budget_control.py \
  --config "$EXP_CONTROL/08_summarize_current_k6_control.yaml" --dry-run

python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_label_budget_control.py \
  --config "$EXP_CONTROL/08_summarize_current_k6_control.yaml"
```

Review the producer-owned lightweight result file set and `git diff` before
committing it.

The final control status is one of `CONTROL_READY_POSITIVE`,
`CONTROL_READY_MIXED`, `CONTROL_READY_WITH_DRIFT`, or
`BLOCKED_CONTROL_CONTRACT`. Historical M1 remains a report reference; the
multi-head handoff fixes this current K=6 control as the primary baseline.
