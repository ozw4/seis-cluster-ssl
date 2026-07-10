# F3 Strat-HMM Milestone-1 Guardrails

This suite separates gains from the structured HMM pretext signal from gains
that can be explained by generic top-block adaptation or accidental
pseudo-label regularization. It compares the existing MAE baseline and the
milestone-1 candidate with two isolated guardrails.

## Questions answered

The **distillation-only** guardrail starts the student from the same MAE
checkpoint, unfreezes the same top encoder block, and retains feature
distillation while setting both prototype and prototype-usage loss weights to
zero. It answers: *is the candidate improvement only top-block continuation or
regularized adaptation?*

The **shuffled-HMM** guardrail retains the pseudo-target artifact schema,
valid-token mask, global label histogram, and confidence values, but assigns
the label/confidence pairs to valid tokens using one deterministic global
shuffle. It answers: *does the ordered spatial/stratigraphic assignment matter?*
The shuffle seed is `188`; repeated builds must produce byte-equivalent arrays
and metadata apart from explicitly recorded provenance fields.

Neither guardrail is a new final predictor. HMM labels remain pretext targets,
and downstream evaluation uses frozen encoder embeddings and the same
`linear_balanced_v1` lithology probe contract as milestone 1.

## Contracts and execution state

The files are ordered by the intended workflow:

1. `01_train_distillation_only_smoke.yaml` and
   `02_train_distillation_only_full.yaml` use the existing pretext trainer. The
   zero `prototype_weight` and `usage_weight` are the distillation-only
   contract.
2. `03_extract_distillation_only_embeddings.yaml`,
   `04_build_distillation_only_token_dataset.yaml`,
   `05_train_distillation_only_probe.yaml`, and
   `06_build_distillation_only_report.yaml` run the milestone-1 downstream
   workflow under the isolated distillation-only model tag.
3. `03_build_shuffled_hmm_pseudo_targets.yaml` defines deterministic shuffle
   inputs, preservation requirements, seed, and an isolated output root.
4. `04_train_shuffled_hmm_full.yaml` is the matching training contract.
5. `05_extract_guardrail_embeddings.yaml`,
   `06_build_guardrail_token_datasets.yaml`, and
   `07_run_guardrail_probes.yaml` define the paired downstream routing and
   isolated roots for both guardrails.
6. `08_summarize_guardrails.yaml` compares the five primary metrics:
   `macro_f1`, `mean_iou`, `balanced_accuracy`, `accuracy`, and `weighted_f1`.

This issue intentionally adds contracts, not the shuffled-target builder or
paired downstream runners. Those mechanics are implemented by the subsequent
guardrail prompts. The distillation-only configs reuse mechanics already
present in the trainer.

## Distillation-only runbook

Validate the smoke config without writing artifacts, then run its two CPU
steps:

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config experiments/f3/facies_benchmark_v1/83_strat_hmm_m1_guardrails/01_train_distillation_only_smoke.yaml \
  --dry-run --device cpu --max-steps 2

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config experiments/f3/facies_benchmark_v1/83_strat_hmm_m1_guardrails/01_train_distillation_only_smoke.yaml \
  --device cpu --max-steps 2
```

Run the full guardrail training:

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config experiments/f3/facies_benchmark_v1/83_strat_hmm_m1_guardrails/02_train_distillation_only_full.yaml
```

Extract embeddings and run the same token-dataset, probe, and report stages as
milestone 1:

```bash
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config experiments/f3/facies_benchmark_v1/83_strat_hmm_m1_guardrails/03_extract_distillation_only_embeddings.yaml

python proc/seis_ssl_cluster/build_f3_lithology_token_dataset.py \
  --config experiments/f3/facies_benchmark_v1/83_strat_hmm_m1_guardrails/04_build_distillation_only_token_dataset.yaml

python proc/seis_ssl_cluster/train_f3_lithology_probe.py \
  --config experiments/f3/facies_benchmark_v1/83_strat_hmm_m1_guardrails/05_train_distillation_only_probe.yaml

python proc/seis_ssl_cluster/build_f3_lithology_report.py \
  --config experiments/f3/facies_benchmark_v1/83_strat_hmm_m1_guardrails/06_build_distillation_only_report.yaml
```

The summary is runnable now:

```bash
python proc/seis_ssl_cluster/summarize_f3_strat_hmm_m1_guardrails.py \
  --config experiments/f3/facies_benchmark_v1/83_strat_hmm_m1_guardrails/08_summarize_guardrails.yaml
```

With `suite.strict: false`, missing model metrics or configured robustness JSON
artifacts are recorded as `pending`. With `suite.strict: true`, any configured
missing artifact is an error. Label-budget and split/index summaries are
included when their optional JSON paths are configured and available.

## Artifact isolation

Stable model tags are:

- `strat_hmm_m1_guardrail_distill_only`
- `strat_hmm_m1_guardrail_shuffled_hmm`

Their pretraining, embedding, and lithology outputs live under their own model
tag components. The shuffled pseudo-target root and the summary report root are
also separate. None use the candidate root
`strat_hmm_pretext_m1_k6_topblock1_distill`, so guardrail runs cannot overwrite
milestone-1 candidate outputs.

Lateral smoothing, multi-resolution heads, HMM-map-as-final-output evaluation,
and the optional later guardrails are outside this suite.
