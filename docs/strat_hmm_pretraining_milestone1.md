# Strat HMM Pretraining Milestone 1

Milestone 1 uses stratigraphic HMM clustering as a structured pretext task for
F3 encoder adaptation. HMM labels are pseudo-targets for representation learning;
they are not final lithology labels and are not evaluated as final task output.

## Required Inputs

- An existing MAE checkpoint compatible with the F3 embedding workflow.
- Existing F3 embedding artifacts from `20_embedding`.
- Existing `stratigraphic_hmm_kmeans` clustering output from
  `60_stratigraphic_clustering`.
- F3 amplitude manifest, path list, registered seismic volume, and lithology
  label artifacts from the existing F3 preparation and lithology workflows.

The milestone configs live in:

```text
experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1/
```

## Commands

Export bootstrap HMM labels to pseudo-target artifacts:

```bash
bash experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1/01_export_bootstrap_hmm_pseudo_targets.sh
```

Run the CPU smoke training check:

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

Run the full single-head top-block distillation pretext training:

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1/03_train_single_head_topblock_distill_full.yaml
```

Extract student embeddings:

```bash
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1/04_extract_student_embeddings.yaml
```

Run the existing few-label lithology probe and report:

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

Optionally dry-run pseudo-target refresh from smoke prototype logits:

```bash
python proc/seis_ssl_cluster/build_strat_hmm_pseudo_targets.py \
  --config experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1/08_refresh_pseudo_targets_from_logits_smoke.yaml \
  --dry-run \
  --device cpu \
  --overwrite
```

## Expected Outputs

Bootstrap pseudo-target exports contain labels, confidence, valid-token masks,
and JSON metadata under the configured pseudo-target root. Strat HMM pretext
training writes `latest.pt` and `best.pt` checkpoints under the configured
pretraining output root.

The strat HMM checkpoints are compatible with the existing embedding extraction
command. Extraction loads the student encoder state and ignores the ordered
prototype head when writing embeddings.

## Evaluation

Evaluate the student only through downstream few-label F3 lithology probe
metrics, especially macro F1 and mean IoU. Compare against:

- existing MAE encoder baseline
- random encoder baseline
- z-only baseline
- xyz-coordinate baseline

The HMM label map can be inspected for debugging, but it must not be promoted to
the final lithology prediction.

## Guardrails

Track or run these guardrails before expanding the method:

- z-only HMM target
- shuffled HMM target
- no-HMM prototype-only training
- distillation-only training
- MAE continuation or top-block-only baseline, when available

If HMM maps appear visually better but downstream macro F1 or mean IoU does not
improve over the existing baselines, stop this direction for milestone 1. Do not
proceed to lateral smoothing, multi-resolution heads, or HMM-map-as-final-output
evaluation.
