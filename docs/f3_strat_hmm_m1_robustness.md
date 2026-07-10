# F3 Strat-HMM M1 Robustness Suites

This contract defines the next F3 strat-HMM milestone-1 robustness layer under:

```text
/workspace/artifacts/seis_ssl_cluster/lithology/f3/facies_benchmark_v1/robustness/
```

It has two suites:

- `label_budget_m1_v1`: compares the existing MAE encoder with `strat_hmm_pretext_m1_k6_topblock1_distill` under matched small-label training subsets.
- `split_index_m1_v1`: compares the same encoders under alternative train/validation slice assignments.

The comparison is paired. For each budget or split condition, both encoders must use the same token rows, labels, validation rows, probe hyperparameters, and probe random state. Reports should center the paired deltas:

```text
delta_macro_f1 = strat_hmm - mae
delta_mean_iou = strat_hmm - mae
delta_balanced_accuracy = strat_hmm - mae
delta_accuracy = strat_hmm - mae
```

Each suite writes:

```text
suite_config_resolved.json
suite_manifest.json
reports/
  paired_metrics.csv
  paired_deltas.csv
  summary.md
```

Individual label-budget runs may live under `model=<model_tag>/budget=<budget_id>/subsample_seed=<seed>/`. Individual split-index runs may live under `split=<split_id>/model=<model_tag>/`.

`latest.pt` versus `best.pt` checkpoint selection is out of scope for these suites because this contract measures robustness to labels and split/index assignment, not checkpoint policy. Probe seed sweeps are also out of scope because M1 requires a paired comparison with the same probe random state inside each condition, not an estimate of probe seed noise.

Expected run order:

1. Resolve and write the suite config and manifest.
2. Materialize the paired label-budget subsets or split/index assignments.
3. Run the MAE baseline and strat-HMM candidate with matched probe settings for each condition.
4. Aggregate `paired_metrics.csv`, `paired_deltas.csv`, and `summary.md`.

## Final decision

Both robustness suites are **Go**.

- Label-budget robustness shows larger gains in low-label regimes. At `cap25`,
  `delta_macro_f1=+0.053841` and `delta_mean_iou=+0.054076`.
- Split/index robustness has positive macro F1 and mean IoU deltas on every
  tested split.

This is strong F3 robustness evidence, but its scope is not cross-survey
generalization. Full-budget balanced accuracy on the original split is lower
(`delta=-0.012804`), and class 5 Zechstein and class 3 Rijnland/Chalk remain
monitoring items. The HMM maps are diagnostic pretext artifacts, not final
lithology labels or final evaluated outputs.

After both suites, regenerate and publish the consolidated result:

```bash
python proc/seis_ssl_cluster/summarize_f3_strat_hmm_m1_results.py \
  --config experiments/f3/facies_benchmark_v1/82_strat_hmm_m1_results/01_summarize_m1_results.yaml
```

The next step is guardrail validation, not immediate method expansion. Use the
runbook in `docs/f3_strat_hmm_m1_guardrails.md`.
