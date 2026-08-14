# F3 Strat-HMM milestone-1 pretraining guardrail producers

This directory retains two scientific pretraining controls: a
distillation-only student and a deterministic shuffled-HMM target student.
They preserve the milestone-1 data, initialization, and training geometry while
isolating whether adaptation or ordered pseudo-target structure drives the
pretext behavior.

Active stages are:

1. `01` and `02`: distillation-only smoke and full pretraining.
2. `03_build_shuffled_hmm_pseudo_targets`: deterministic shuffled-target
   generation.
3. `03_extract_distillation_only_embeddings`: distillation-only extraction.
4. `05_extract_guardrail_embeddings`: shared extraction validation.
5. `07_train_shuffled_hmm_smoke`, `08_train_shuffled_hmm_full`, and
   `09_extract_shuffled_hmm_embeddings`: shuffled-target pretraining and
   extraction.

Complete outputs remain under `artifacts/seis_ssl_cluster/`. Token probes,
voxel-count label budgets, seed aggregation, downstream summaries, and report
publication are retired.
