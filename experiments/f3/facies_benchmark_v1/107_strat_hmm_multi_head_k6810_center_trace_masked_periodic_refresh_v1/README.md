# F3 center-trace masked periodic HMM refresh

Experiment 107 is the periodic student-embedding/HMM-center refresh treatment
defined by [the periodic refresh contract](../../../../docs/f3_center_trace_masked_periodic_hmm_refresh_plan.md).
It preserves the experiment-104 center-trace scientific fields and changes
only the periodic refresh identity, generation/output roots, and final
checkpoint selection policy.

Set the workspace paths and the exact immutable initial target digest first:

```bash
export SEIS_SSL_CLUSTER_WORKSPACE=/workspace
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/workspace/artifacts/seis_ssl_cluster
export SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256="$(sha256sum \
  "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT"/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1/multi_head_target_manifest.json \
  | awk '{print $1}')"
export EXP=experiments/f3/facies_benchmark_v1/107_strat_hmm_multi_head_k6810_center_trace_masked_periodic_refresh_v1
```

The exact execution order is:

```bash
python proc/seis_ssl_cluster/validate_f3_center_trace_masked_periodic_refresh.py \
  --config "$EXP/04_validate_periodic_refresh_pretraining.yaml" --phase inputs

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/01_train_periodic_refresh_smoke.yaml" --dry-run --device cpu --max-steps 2
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/01_train_periodic_refresh_smoke.yaml" --device cpu --max-steps 2

python proc/seis_ssl_cluster/validate_f3_center_trace_masked_periodic_refresh.py \
  --config "$EXP/04_validate_periodic_refresh_pretraining.yaml" --phase smoke
```

The smoke root is isolated from the full root. The CLI `--device cpu
--max-steps 2` overrides are runtime-only; the YAML keeps the full scientific
identity, 25 epochs, 4096 samples per epoch, batch size 4, and exact refresh
schedule `[2, 5, 8, 11, 14, 17, 20]`. Two partial steps remain in generation
zero, so no refreshed generation is created and no smoke checkpoint is a
resume source for the full run.

An existing partial or foreign owned output is rejected by default. Quarantine
is explicit and moves the selected sibling to a timestamped path:

```bash
python proc/seis_ssl_cluster/validate_f3_center_trace_masked_periodic_refresh.py \
  --config "$EXP/04_validate_periodic_refresh_pretraining.yaml" \
  --phase smoke --quarantine-invalid
```

After an authorized full run, resume only from the full root's `latest.pt`.
The full validator requires eight complete generations, schema-8
`latest.pt`/`selected.pt`, no `best.pt`, and selection policy
`final_completed_epoch_v1`:

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/02_train_periodic_refresh_full.yaml"
python proc/seis_ssl_cluster/validate_f3_center_trace_masked_periodic_refresh.py \
  --config "$EXP/04_validate_periodic_refresh_pretraining.yaml" --phase checkpoints
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/03_extract_periodic_refresh_embeddings.yaml"
python proc/seis_ssl_cluster/validate_f3_center_trace_masked_periodic_refresh.py \
  --config "$EXP/04_validate_periodic_refresh_pretraining.yaml" --phase complete

python proc/seis_ssl_cluster/publish_f3_center_trace_masked_periodic_refresh_results.py \
  --config "$EXP/05_review_periodic_refresh_results.yaml" --dry-run
python proc/seis_ssl_cluster/publish_f3_center_trace_masked_periodic_refresh_results.py \
  --config "$EXP/05_review_periodic_refresh_results.yaml"
```

The final embedding must be unmasked and bound to the completed selected
checkpoint, final target generation, fixed preprocessing, and valid-token
mask. A PASS handoff is published only by `complete`; `inputs`, `smoke`, and
`checkpoints` never publish it. Exact PASS handoffs are reused without a
content or mtime change. A stale handoff requires `--quarantine-invalid`.
The publication dry-run and write command are run only after `complete` has
published and validated the PASS handoff; they publish the allowlisted,
lightweight review tree and do not copy raw checkpoints or embeddings.

This issue intentionally executes only the live `inputs` phase, smoke dry-run,
isolated CPU two-step smoke, and smoke validation. Full 25-epoch pretraining,
the seven real refreshes, final embedding extraction, original-split decoder
jobs, and six-split jobs are executed zero times here.
