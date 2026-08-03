# F3 center-trace masked schema-7 pretraining

Experiment 104 is the hard-label center-trace masked successor to the frozen
`mh_nocons` baseline. It keeps the K=6/8/10 target manifest, MAE
initialization, model geometry, optimizer settings, top-block freeze, and
distillation settings unchanged. Its fixed scientific identity is
`ctmask010_nocons`: 10% of eligible valid XY columns are selected with the
full-Z mask semantics, replaced by an independently seeded learned encoder
token, and trained with the masked/visible 0.50/0.50 supervised objective.
Consistency remains disabled. Checkpoints must use schema 7.

Set the artifact root and the exact immutable target digest from the workspace
root:

```bash
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/path/to/artifacts/seis_ssl_cluster
export EXP=experiments/f3/facies_benchmark_v1/104_strat_hmm_multi_head_k6810_center_trace_masked_v1
export SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256="$(sha256sum \
  "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT"/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1/multi_head_target_manifest.json \
  | awk '{print $1}')"
```

For issue #305, run only the `inputs` and CPU two-step `smoke` phases below.
Full pretraining, embedding extraction, checkpoint validation, and complete
handoff publication are intentionally deferred; this issue must not create or
modify the full checkpoint or embedding roots.

The `inputs` phase atomically records the pre-execution Git commit, dirty
status, and binary diff digest in
`$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/f3/facies_benchmark_v1/.f3_center_trace_masked_pretraining_execution.json`.
The `smoke` phase records the corresponding after-state. Later phases reject a
missing or stale sidecar instead of reusing execution evidence from another
config or target.

Successful phases also publish read-only validation reports beside that
sidecar: `.f3_center_trace_masked_pretraining_inputs.json` fingerprints the
real amplitude manifest, volume, normalization, teacher, and student inputs;
`.f3_center_trace_masked_pretraining_smoke.json` binds the real-data CPU
two-step schema-7 checkpoint, metrics, and input evidence. Neither report is a
PASS handoff.

```bash
python proc/seis_ssl_cluster/validate_f3_center_trace_masked_pretraining.py \
  --config "$EXP/04_validate_center_trace_masked_pretraining.yaml" --phase inputs

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/01_train_center_trace_masked_smoke.yaml" --dry-run --device cpu --max-steps 2
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/01_train_center_trace_masked_smoke.yaml" --device cpu --max-steps 2
python proc/seis_ssl_cluster/validate_f3_center_trace_masked_pretraining.py \
  --config "$EXP/04_validate_center_trace_masked_pretraining.yaml" --phase smoke
```

The smoke YAML intentionally keeps `train.max_steps: null` in both the
scientific identity and runtime config. The required `--max-steps 2` CLI
override is applied after resolution and therefore does not create a new
scientific identity.

The full and extraction YAML files, plus the `checkpoints` and `complete`
validator phases, are retained for a later authorized continuation. Do not run
those phases in issue #305.

Resume and quarantine procedure:

- The smoke output root is isolated from the full output root and cannot resume
  from it. If the isolated smoke root is foreign or partial, validate it with
  `--quarantine-invalid` before rerunning the two-step smoke. The validator
  moves that root to a timestamped `.quarantine.*` sibling and never changes
  the full root:

```bash
python proc/seis_ssl_cluster/validate_f3_center_trace_masked_pretraining.py \
  --config "$EXP/04_validate_center_trace_masked_pretraining.yaml" \
  --phase smoke --quarantine-invalid
```

- A later authorized full run may resume only from its own `latest.pt`; it must
  never resume from the isolated smoke root. A stale, partial, or invalid
  handoff must be preserved with `--quarantine-invalid` before a later
  `complete` phase publishes a replacement. Those full-run and extraction
  commands are intentionally omitted here because they are not executed in
  issue #305.

The future handoff path is
`preflight/center_trace_masked_handoff.json` under the full center-trace
checkpoint root. It is not published by issue #305. No downstream decoder or
six-split result is part of this experiment root.
