# F3 Strat-HMM Pretraining Milestone 1

## Final decision

Milestone 1 passed.

The strat-HMM pretext student improves over the existing MAE baseline on the
single full-budget F3 lithology probe, with accuracy increasing from `0.886477`
to `0.896330` and mean IoU increasing from `0.650059` to `0.660868`.

Label-budget robustness is a **Go**. The gains are larger in low-label regimes;
at `cap25`, `delta_macro_f1` is `+0.053841` and `delta_mean_iou` is
`+0.054076`.

Split/index robustness is also a **Go**. Macro F1 and mean IoU deltas are
positive on every tested split.

The method should proceed to guardrail validation and then, if the guardrails
pass, to next-stage method extensions.

## Result strength and evidence scope

The result strength is a consistent positive downstream signal: the original
full-budget probe improves on accuracy and mean IoU, the largest gains occur
under small label budgets, and every tested split improves on macro F1 and mean
IoU.

The evidence scope remains limited to F3 and the frozen-embedding lithology
probe contract used here. F3-only evidence does not establish cross-survey
generalization. In addition:

- full-budget balanced accuracy on the original split decreases from
  `0.843767` to `0.830964` (`delta=-0.012804`);
- class 5 Zechstein and class 3 Rijnland/Chalk remain monitoring items; and
- the HMM label maps are diagnostic structured pretext targets, not final
  lithology labels or final evaluated outputs.

The published summary is under
`reports/f3/facies_benchmark_v1/strat_hmm_pretext_m1/`.

## Reproducible command order

Paths in the configs are explicit defaults rooted at `/workspace`; edit them
before running when the artifact or data root differs.

1. Run the candidate workflow in
   `experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1/README.md`.
2. Run label-budget robustness and then split/index robustness in
   `experiments/f3/facies_benchmark_v1/81_strat_hmm_m1_robustness/README.md`.
3. Regenerate and publish the consolidated milestone-1 result.
4. Run the guardrails in
   `experiments/f3/facies_benchmark_v1/83_strat_hmm_m1_guardrails/README.md`.
5. If the guardrails pass, begin next-stage method extensions.

Validate the summary inputs and output paths without writing:

```bash
python proc/seis_ssl_cluster/summarize_f3_strat_hmm_m1_results.py \
  --config experiments/f3/facies_benchmark_v1/82_strat_hmm_m1_results/01_summarize_m1_results.yaml \
  --dry-run
```

Regenerate the milestone-1 summary and publish its small result artifacts:

```bash
python proc/seis_ssl_cluster/summarize_f3_strat_hmm_m1_results.py \
  --config experiments/f3/facies_benchmark_v1/82_strat_hmm_m1_results/01_summarize_m1_results.yaml
```

The `publish.enabled: true` block in that config copies the summary, tables,
figures to the repository-relative report directory. Set
the explicit `outputs.output_dir` and `publish.output_dir` in the YAML to the
desired editable paths before running.
