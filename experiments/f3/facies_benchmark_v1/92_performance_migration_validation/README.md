# Performance migration validation

This experiment validates the current performance-improved implementation
against the historical M3-V-LB baseline without retraining scientific models or
overwriting historical artifacts. It is limited to checkpoint loading, M1
embedding extraction, an existing M1 linear-probe prediction replay, a K=6
Stratigraphic HMM replay, pseudo-target parity, and the synthetic performance
benchmark.

The fixed current revision is
`332478be21a021e46ee6c1d9423f14859b0cd819`; the historical baseline is
`7731f341a293ea0c5cb5c5dfabba574148861e3a`. All generated data is isolated
under:

```text
/workspace/artifacts/seis_ssl_cluster/migration_validation/f3/
facies_benchmark_v1/main_332478be/
```

Historical checkpoints, overlap-x16 embeddings, HMM outputs, pseudo-targets,
linear probe, scaler, token dataset, M3-V, and M3-V-LB artifacts are read
only. A pre-existing incomplete or mismatched migration artifact is quarantined
by the migration runner; it is never removed or overwritten in place.

## Scientific contract

M1 extraction holds the historical scientific settings fixed: window
`[128, 128, 128]`, overlap `[112, 64, 64]`, float16 output, batch size 1,
minimum token-valid fraction 0.5, AMP disabled, and no prefetching. The two
new outputs differ only in runtime cache mode:

- `02_extract_m1_cache_off.yaml`: preprocessing cache `off`.
- `03_extract_m1_cache_memmap.yaml`: persistent memmap cache with 16-x chunks.

The original M1 checkpoint predates the explicit `finite_check_mode` field.
The migration wrapper records the source-code evidence for the historical
`off` behavior and injects it only into the in-memory checkpoint-owned
extraction contract. The ordinary embedding YAMLs deliberately do not override
checkpoint-owned preprocessing fields. This prevents a silent current default
from being mistaken for the legacy scientific contract.

The HMM replay uses the historical **MAE** embedding source (not the M1
embedding), K=6, L2 normalization, token-phase residualization, PCA-64, the
ordered transition/path-prior configuration, and seed 42. It writes only to
the migration root. The pseudo-target export is legacy-compatible schema v1:
no boundary-weight field is emitted for M1.

`reports/f3/facies_benchmark_v1/performance_migration_validation/` receives
only the completed Markdown, JSON, CSV, small PNG files, and README.
Checkpoints, arrays, joblib objects, embeddings, clustering labels,
pseudo-target arrays, prepared-feature caches, temporary directories, and
quarantines are prohibited from that directory.

## Execution order

```bash
cd /workspace
export PYTHONPATH="/workspace/src${PYTHONPATH:+:$PYTHONPATH}"
export EXP_MIG=experiments/f3/facies_benchmark_v1/92_performance_migration_validation

# 1. Repository state and machine-readable input inventory.
python proc/seis_ssl_cluster/validate_performance_migration.py \
  --config "$EXP_MIG/01_checkpoint_smoke.yaml" --stage preflight

# 2. Strict CPU checkpoint load and deterministic crop smoke.
python proc/seis_ssl_cluster/validate_performance_migration.py \
  --config "$EXP_MIG/01_checkpoint_smoke.yaml" --stage checkpoint-smoke

# 3-6. Validate the ordinary extraction contracts, then run staging-safe
# extraction with the evidenced legacy finite-check compatibility adapter.
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP_MIG/02_extract_m1_cache_off.yaml" --dry-run
python proc/seis_ssl_cluster/validate_performance_migration.py \
  --config "$EXP_MIG/01_checkpoint_smoke.yaml" \
  --stage extract-m1 --embedding-config "$EXP_MIG/02_extract_m1_cache_off.yaml" \
  --only-missing
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP_MIG/03_extract_m1_cache_memmap.yaml" --dry-run
python proc/seis_ssl_cluster/validate_performance_migration.py \
  --config "$EXP_MIG/01_checkpoint_smoke.yaml" \
  --stage extract-m1 --embedding-config "$EXP_MIG/03_extract_m1_cache_memmap.yaml" \
  --only-missing

# 7-9. A/B/C embedding diagnostics and an existing-probe prediction replay.
python proc/seis_ssl_cluster/validate_performance_migration.py \
  --config "$EXP_MIG/01_checkpoint_smoke.yaml" --stage embedding-parity
python proc/seis_ssl_cluster/validate_performance_migration.py \
  --config "$EXP_MIG/01_checkpoint_smoke.yaml" --stage probe-parity
python proc/seis_ssl_cluster/validate_performance_migration.py \
  --config "$EXP_MIG/01_checkpoint_smoke.yaml" --stage reconstruct-hmm-config

# 10-12. Check the regular clustering config before a staging-safe current-code
# K=6 replay and compare its labels/centers/diagnostics with history.
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config "$EXP_MIG/04_replay_m1_k6_hmm.yaml" --dry-run
python proc/seis_ssl_cluster/validate_performance_migration.py \
  --config "$EXP_MIG/01_checkpoint_smoke.yaml" \
  --stage replay-hmm --hmm-config "$EXP_MIG/04_replay_m1_k6_hmm.yaml" \
  --only-missing
python proc/seis_ssl_cluster/validate_performance_migration.py \
  --config "$EXP_MIG/01_checkpoint_smoke.yaml" --stage hmm-parity

# 13-14. Legacy-compatible M1 pseudo-target export and parity.
python proc/seis_ssl_cluster/validate_performance_migration.py \
  --config "$EXP_MIG/05_export_m1_k6_pseudo_targets.yaml" --stage export-pseudo-targets \
  --only-missing
python proc/seis_ssl_cluster/validate_performance_migration.py \
  --config "$EXP_MIG/01_checkpoint_smoke.yaml" --stage pseudo-target-parity

# 15-16. Synthetic performance diagnostics; no benchmark ratio is reported
# unless the baseline is demonstrably comparable.
python tools/benchmark_seis_ssl_cluster_performance.py --smoke
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_NUM_THREADS=1 \
  python tools/benchmark_seis_ssl_cluster_performance.py \
    --seed 248 --warm-up 3 --repeat 20 \
    --output-json /workspace/artifacts/seis_ssl_cluster/migration_validation/f3/facies_benchmark_v1/main_332478be/benchmark/current.json \
    --output-markdown /workspace/artifacts/seis_ssl_cluster/migration_validation/f3/facies_benchmark_v1/main_332478be/benchmark/current.md

# 17-19. Calculate the ordered migration decision, publish only light files,
# and write the handoff.
python proc/seis_ssl_cluster/validate_performance_migration.py \
  --config "$EXP_MIG/06_summarize_performance_migration.yaml" --stage summarize --dry-run
python proc/seis_ssl_cluster/validate_performance_migration.py \
  --config "$EXP_MIG/06_summarize_performance_migration.yaml" --stage summarize
python proc/seis_ssl_cluster/validate_performance_migration.py \
  --config "$EXP_MIG/06_summarize_performance_migration.yaml" --stage publish
```

Review the producer-owned lightweight result file set and `git diff` before
committing published outputs.

The decision is constrained by the recorded artifacts. `PASS_REUSE_EXISTING`
requires exact checkpoint smoke, valid-token, probe prediction/confusion,
K=6 label, and pseudo-target label contracts. Small embedding or center drift
can only become `PASS_WITH_NUMERIC_DRIFT` when downstream predictions and
labels remain exact and no confidence threshold is crossed. HMM or
pseudo-target label drift requires `REBUILD_M1_REQUIRED`; a probe-only change
requires `REEXTRACT_REQUIRED`; any numerical-contract failure blocks further
experiments.

The resulting handoff states the appropriate multi-head baseline policy. No
six-split evaluation, voxel decoder training, M3-V/M3-V-LB rerun, multi-head
experiment, or scientific parameter change is in scope here.
