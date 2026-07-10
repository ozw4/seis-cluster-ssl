# F3 Strat-HMM Milestone-1 Guardrails

Milestone 1 passed its main, label-budget, and split/index decisions. Guardrails
are the required next step before method expansion. They test whether the gain
depends on the structured HMM assignment rather than generic top-block
adaptation or accidental pseudo-label regularization.

HMM maps remain diagnostic pretext artifacts. The evaluated output is the
standard frozen-encoder `linear_balanced_v1` F3 lithology probe, not an HMM
pseudo-label map.

All default paths are explicit under `/workspace`. Edit the YAML paths when the
artifact or data root differs. The complete contracts and artifact locations
are documented in
`experiments/f3/facies_benchmark_v1/83_strat_hmm_m1_guardrails/README.md`.

## Distillation-only guardrail

Run training, embedding extraction, token-dataset construction, the matched
probe, and its report in that order:

```bash
export EXP=experiments/f3/facies_benchmark_v1/83_strat_hmm_m1_guardrails

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/02_train_distillation_only_full.yaml"
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/03_extract_distillation_only_embeddings.yaml"
python proc/seis_ssl_cluster/build_f3_lithology_token_dataset.py \
  --config "$EXP/04_build_distillation_only_token_dataset.yaml"
python proc/seis_ssl_cluster/train_f3_lithology_probe.py \
  --config "$EXP/05_train_distillation_only_probe.yaml"
python proc/seis_ssl_cluster/build_f3_lithology_report.py \
  --config "$EXP/06_build_distillation_only_report.yaml"
```

## Shuffled-HMM guardrail

Build the deterministic shuffled pseudo-targets first, then run the matched
training and downstream evaluation stages:

```bash
export EXP=experiments/f3/facies_benchmark_v1/83_strat_hmm_m1_guardrails

python proc/seis_ssl_cluster/shuffle_strat_hmm_pseudo_targets.py \
  --config "$EXP/03_build_shuffled_hmm_pseudo_targets.yaml"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/08_train_shuffled_hmm_full.yaml"
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/09_extract_shuffled_hmm_embeddings.yaml"
python proc/seis_ssl_cluster/build_f3_lithology_token_dataset.py \
  --config "$EXP/10_build_shuffled_hmm_token_dataset.yaml"
python proc/seis_ssl_cluster/train_f3_lithology_probe.py \
  --config "$EXP/11_train_shuffled_hmm_probe.yaml"
python proc/seis_ssl_cluster/build_f3_lithology_report.py \
  --config "$EXP/12_build_shuffled_hmm_report.yaml"
```

## Paired low-budget guardrails

Run the existing label-budget pipeline for each guardrail. Both suites use
`cap25`, `cap100`, `cap500`, and `full`, with the milestone-1 baseline and the
same subsample seeds so pairing can be verified from the suite manifests.

```bash
export EXP=experiments/f3/facies_benchmark_v1/83_strat_hmm_m1_guardrails
export ROOT=/workspace/artifacts/seis_ssl_cluster/lithology/f3/facies_benchmark_v1/guardrails/strat_hmm_m1_guardrails_v1/label_budget

python proc/seis_ssl_cluster/build_f3_lithology_label_budget_datasets.py \
  --config "$EXP/14_build_distillation_only_label_budget_datasets.yaml"
python proc/seis_ssl_cluster/run_f3_lithology_label_budget_probes.py \
  --config "$EXP/15_run_distillation_only_label_budget_probes.yaml" \
  --only-missing
python proc/seis_ssl_cluster/summarize_f3_lithology_label_budget_robustness.py \
  --suite-root "$ROOT/distillation_only"

python proc/seis_ssl_cluster/build_f3_lithology_label_budget_datasets.py \
  --config "$EXP/16_build_shuffled_hmm_label_budget_datasets.yaml"
python proc/seis_ssl_cluster/run_f3_lithology_label_budget_probes.py \
  --config "$EXP/17_run_shuffled_hmm_label_budget_probes.yaml" \
  --only-missing
python proc/seis_ssl_cluster/summarize_f3_lithology_label_budget_robustness.py \
  --suite-root "$ROOT/shuffled_hmm"
```

## Summarize guardrails

Validate the summary routing, then write the four-model comparison:

```bash
python proc/seis_ssl_cluster/summarize_f3_strat_hmm_m1_guardrails.py \
  --config experiments/f3/facies_benchmark_v1/83_strat_hmm_m1_guardrails/13_summarize_guardrails.yaml \
  --dry-run

python proc/seis_ssl_cluster/summarize_f3_strat_hmm_m1_guardrails.py \
  --config experiments/f3/facies_benchmark_v1/83_strat_hmm_m1_guardrails/13_summarize_guardrails.yaml
```

The final summary reads the low-budget pipeline's `summary_by_budget.csv` and
`suite_manifest.json` directly. With `suite.strict: false`, absent guardrail
artifacts are reported as `pending`. Use `suite.strict: true` for the final
decision so missing metrics or verified `cap25`, `cap100`, `cap500`, and `full`
comparisons fail rather than being mistaken for evidence. Proceed to method
extensions only after both guardrail results are complete and support the
structured-HMM interpretation.
