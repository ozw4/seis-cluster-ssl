# F3 random-init self-target HMM K=6 distill 0.1

This experiment defines the F3 Condition 6a source. A seed-42 random encoder is
used for the ordered HMM K=6 pseudo-targets, frozen teacher, and student
initialization. The Single-Head HMM stage runs for 25 epochs with distillation
weight 0.1, then the frozen encoder is evaluated on the canonical v3 matrix of
five layouts by three supervision sizes.

The versioned YAML is authoritative. From the repository root, set the shared
[F3 environment](../README.md), then run each writing command only after its
supported `--dry-run` succeeds:

```bash
export EXP=experiments/f3/facies_benchmark_v2/125_lithology_random_init_self_target_hmm_k6_distill010_v1

python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/10_hmm_targets/random_init/01_extract_embeddings.yaml"
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config "$EXP/10_hmm_targets/random_init/k6/02_cluster_hmm_k6.yaml"
bash "$EXP/10_hmm_targets/random_init/k6/03_export_pseudo_targets.sh"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/20_stage2/random_init/hmm/k6/01_gpu_feasibility_1step.yaml"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/20_stage2/random_init/hmm/k6/02_full_25ep.yaml"
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/30_embeddings/01_extract_random_init_self_target_hmm_k6_distill010.yaml"
```

Run the 15 downstream cells through
[`40_downstream/01_candidate.yaml`](40_downstream/01_candidate.yaml) with
`run_f3_lithology_candidate.py`, then use
`summarize_f3_lithology_candidate.py`. The cross-condition frozen result table
is owned by the [HMM_v1 freeze](../../../../reports/hmm_v1/README.md); this
experiment directory does not duplicate those numbers.

If HMM training was interrupted, resume only from this full run's own
`latest.pt`. Never use the smoke or initialization checkpoint as resume state.
