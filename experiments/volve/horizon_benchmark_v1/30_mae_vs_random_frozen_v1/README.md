# Volve frozen MAE versus random horizon benchmark

This suite trains the shared five-horizon decoder on frozen Volve MAE or
seed-42 random embeddings. The 004 physical layouts and fixed split plan are
reused unchanged, producing 2 model sources × 5 layouts × 3 nested sizes = 30
paired jobs. The decoder is initialized with seed 42000 in every job.

Set the read-only public root and a writable artifact root:

```bash
export SEIS_SSL_CLUSTER_VOLVE_ROOT=/home/dcuser/public_data/field/volve
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/absolute/path/to/artifacts

FROZEN=experiments/volve/horizon_benchmark_v1/30_mae_vs_random_frozen_v1
LAYOUTS=experiments/volve/horizon_benchmark_v1/20_horizon_supervision/01_layouts.yaml
```

Create or verify the canonical registration, then extract the paired embeddings
with the existing generic extractor:

```bash
python proc/seis_ssl_cluster/prepare_volve_canonical_inputs.py --only-missing

python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$FROZEN/01_extract_pretrained_embeddings.yaml" \
  --device cuda \
  --skip-existing

python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$FROZEN/02_extract_random_embeddings.yaml" \
  --device cuda \
  --skip-existing
```

The extraction preflight requires identical architecture, geometry, token-valid
mask, preprocessing, and canonical scientific identity. It also requires
different checkpoint hashes, the completed Volve `full_100ep/latest.pt` role,
and the seed-42 `random_init` role. With `min_token_valid_fraction: 1.0`, every
token touching a missing trace or survey padding is invalid.

Inspect one read-only condition before training:

```bash
python proc/seis_ssl_cluster/run_volve_horizon_frozen.py \
  --config "$FROZEN/03_horizon_frozen.yaml" \
  --model pretrained \
  --layout layout_000 \
  --size small \
  --layout-config "$LAYOUTS" \
  --dry-run
```

Run one job and resume its rolling checkpoint if interrupted:

```bash
python proc/seis_ssl_cluster/run_volve_horizon_frozen.py \
  --config "$FROZEN/03_horizon_frozen.yaml" \
  --model pretrained \
  --layout layout_000 \
  --size small \
  --layout-config "$LAYOUTS" \
  --device cuda

RUN="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/horizon/volve/horizon_benchmark_v1/mae_vs_random_frozen_v1/runs/model=pretrained/layout=layout_000/size=small"

python proc/seis_ssl_cluster/run_volve_horizon_frozen.py \
  --config "$FROZEN/03_horizon_frozen.yaml" \
  --model pretrained \
  --layout layout_000 \
  --size small \
  --layout-config "$LAYOUTS" \
  --device cuda \
  --resume "$RUN/latest.pt"
```

`best.pt` changes only when validation macro MAE is strictly lower. After 50
epochs, the runner reloads `best.pt` and traverses the fixed test tiles once,
writing common-primary and per-horizon-secondary metrics to `metrics.json`.
No probability volume is produced. `latest.pt` contains the complete resume
identity and training position.

After the one-condition smoke and dry-run checks, launch all 30 jobs with the
same command contract (schedule these commands as appropriate for the host):

```bash
for model in pretrained random; do
  for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do
    for size in small medium large; do
      python proc/seis_ssl_cluster/run_volve_horizon_frozen.py \
        --config "$FROZEN/03_horizon_frozen.yaml" \
        --model "$model" \
        --layout "$layout" \
        --size "$size" \
        --layout-config "$LAYOUTS" \
        --device cuda
    done
  done
done
```
