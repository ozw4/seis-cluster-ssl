# F3 strat-HMM M2-A boundary-weighting runbook

M1 is complete and a strong positive result. Do not repeatedly revalidate it.
M2-A asks whether reducing the contribution of tokens close to decoded HMM
transitions improves downstream lithology transfer. M1 is already
confidence-aware: both conditions retain the same confidence weighting, and
M2-A changes only boundary treatment.

The preregistered candidate uses `alpha=0.5`, `tau=2.0`, `k=6`, top-block
unfreezing of one block, distillation weight `0.2`, seed `42`, and the M1 crop,
AGC, zero-mask, architecture, optimizer, batch size, epoch count, probe,
label-budget, and split/index settings. The sole experimental change is the
schema-v2 boundary-weight array used by the prototype loss. Schema-v1 M1
artifacts remain readable and imply the legacy unit weight; schema-v2 records
and validates explicit boundary weighting. No schema-v1 artifact is rewritten.

## Ordered execution and artifacts

Run every command from the repository root, in this order. The first command
of the full M2-A workflow is the alpha-zero parity export:

```bash
bash experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/01_export_alpha0_parity_bootstrap.sh
bash experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/02_export_boundary_weighted_bootstrap.sh
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/03_train_boundary_smoke.yaml --device cpu --max-steps 2
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/04_train_boundary_full.yaml
python proc/seis_ssl_cluster/extract_embeddings.py --config experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/05_extract_student_embeddings.yaml
python proc/seis_ssl_cluster/build_f3_lithology_token_dataset.py --config experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/06_build_lithology_token_dataset.yaml
python proc/seis_ssl_cluster/train_f3_lithology_probe.py --config experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/07_train_lithology_probe.yaml
python proc/seis_ssl_cluster/build_f3_lithology_report.py --config experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/08_build_lithology_report.yaml
python proc/seis_ssl_cluster/build_f3_lithology_label_budget_datasets.py --config experiments/f3/facies_benchmark_v1/85_strat_hmm_m2a_robustness/01_build_label_budget_datasets.yaml
python proc/seis_ssl_cluster/run_f3_lithology_label_budget_probes.py --config experiments/f3/facies_benchmark_v1/85_strat_hmm_m2a_robustness/02_run_label_budget_probes.yaml --only-missing
python proc/seis_ssl_cluster/summarize_f3_lithology_label_budget_robustness.py --config experiments/f3/facies_benchmark_v1/85_strat_hmm_m2a_robustness/03_summarize_label_budget.yaml
python proc/seis_ssl_cluster/build_f3_lithology_split_sweep_datasets.py --config experiments/f3/facies_benchmark_v1/85_strat_hmm_m2a_robustness/04_build_split_sweep_datasets.yaml --only-missing
python proc/seis_ssl_cluster/run_f3_lithology_split_sweep_probes.py --config experiments/f3/facies_benchmark_v1/85_strat_hmm_m2a_robustness/05_run_split_sweep_probes.yaml --only-missing
python proc/seis_ssl_cluster/summarize_f3_lithology_split_robustness.py --config experiments/f3/facies_benchmark_v1/85_strat_hmm_m2a_robustness/06_summarize_split_sweep.yaml
PYTHONPATH=src python proc/seis_ssl_cluster/summarize_f3_strat_hmm_m2_results.py --config experiments/f3/facies_benchmark_v1/86_strat_hmm_m2a_results/01_summarize_m2a_results.yaml
```

Expected artifact roots are:

- alpha0 parity: `/workspace/artifacts/seis_ssl_cluster/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_k6_pca64_resid_token_phase_edge8_expected3_iter10_bootstrap_boundary_a000_t2_parity/k6/`
- weighted export: `/workspace/artifacts/seis_ssl_cluster/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_k6_pca64_resid_token_phase_edge8_expected3_iter10_bootstrap_boundary_a050_t2/k6/`
- smoke checkpoint: `/workspace/artifacts/seis_ssl_cluster/pretraining/f3/facies_benchmark_v1/strat_hmm_pretext_m2a_boundary_a050_t2_k6_topblock1_distill_smoke/`
- full checkpoint: `/workspace/artifacts/seis_ssl_cluster/pretraining/f3/facies_benchmark_v1/strat_hmm_pretext_m2a_boundary_a050_t2_k6_topblock1_distill/`
- embeddings: `/workspace/artifacts/seis_ssl_cluster/embeddings/f3/facies_benchmark_v1/strat_hmm_pretext_m2a_boundary_a050_t2_k6_topblock1_distill/overlap_x16/`
- full probe/report: `/workspace/artifacts/seis_ssl_cluster/lithology/f3/facies_benchmark_v1/strat_hmm_pretext_m2a_boundary_a050_t2_k6_topblock1_distill/overlap_x16/png_slices_segy_labels_v1/`
- label budget: `/workspace/artifacts/seis_ssl_cluster/lithology/f3/facies_benchmark_v1/robustness/label_budget_m2a_boundary_vs_m1_v1/`
- split/index: `/workspace/artifacts/seis_ssl_cluster/lithology/f3/facies_benchmark_v1/robustness/split_index_m2a_boundary_vs_m1_v1/`
- final summary: `/workspace/artifacts/seis_ssl_cluster/lithology/f3/facies_benchmark_v1/reports/strat_hmm_m2a_results/`, published to `results/f3/facies_benchmark_v1/strat_hmm_pretext_m2a_boundary/`.

The detailed alpha0 array-equality check is in the experiment
[README](../experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/README.md).
It must pass before the weighted export is used.

## Failure handling and housekeeping

Stop on an incomplete artifact; do not summarize partial evidence as a result.
Resume idempotent robustness jobs with `--only-missing`. Any token identity,
model-tag, suite-manifest, split-inventory, or condition identity mismatch is a
hard error: correct the upstream artifact rather than pairing different rows.
Use balanced accuracy alongside macro-F1/mean-IoU, and inspect class 3 and 5
F1 specifically; overall accuracy cannot override their registered checks.

To regenerate the M1 structured report from real local artifacts (including
class counts and imbalance ratio loaded from the token datasets), run:

```bash
python proc/seis_ssl_cluster/build_f3_lithology_report.py --config experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1/07_build_lithology_report.yaml
```

To generate and publish the guardrail final summary, first require complete
inputs by setting `suite.strict: true` and set `publish.enabled: true` in
`83_strat_hmm_m1_guardrails/13_summarize_guardrails.yaml`, then run:

```bash
PYTHONPATH=src python proc/seis_ssl_cluster/summarize_f3_strat_hmm_m1_guardrails.py --config experiments/f3/facies_benchmark_v1/83_strat_hmm_m1_guardrails/13_summarize_guardrails.yaml
```

The default publish destination is
`results/f3/facies_benchmark_v1/strat_hmm_m1_guardrails/`; only Markdown, JSON,
CSV, and PNG are allowed. This repository does not contain the full external
guardrail artifacts, so no guardrail summary has been fabricated or published.

M2-B ambiguity-aware confidence, M2-C soft targets, lateral smoothing,
multi-resolution training, and EM refresh are explicitly out of scope.
