# Volve frozen MAE versus Random horizon benchmark

This suite compares frozen Volve MAE embeddings with the matched Random encoder
under the shared horizon split and decoder. Scientific split semantics are
documented in
[`docs/volve_horizon_supervision.md`](../../../../docs/volve_horizon_supervision.md).
The YAML files and runner own sources, settings, output paths, metrics, and
resume validation.

Start with the shared [Volve benchmark environment](../README.md).
Complete the linked [MAE pretraining workflow](../10_pretrain/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/README.md),
which owns canonical-input preparation, before extracting embeddings here.

## Execution order

```bash
export FROZEN="$VOLVE_EXP/30_mae_vs_random_frozen_v1"

python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$FROZEN/01_extract_pretrained_embeddings.yaml" \
  --device cuda --skip-existing
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$FROZEN/02_extract_random_embeddings.yaml" \
  --device cuda --skip-existing
```

Inspect one paired condition before live training:

```bash
python proc/seis_ssl_cluster/run_volve_horizon_frozen.py \
  --config "$FROZEN/03_horizon_frozen.yaml" \
  --model pretrained --layout layout_000 --size small \
  --layout-config "$VOLVE_LAYOUTS" --dry-run
```

Run every model/layout/size cell with the same config. If a cell is interrupted,
resume only that cell from its own rolling checkpoint and with the same runtime
precision.

```bash
for model in pretrained random; do
  for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do
    for size in small medium large; do
      python proc/seis_ssl_cluster/run_volve_horizon_frozen.py \
        --config "$FROZEN/03_horizon_frozen.yaml" \
        --model "$model" --layout "$layout" --size "$size" \
        --layout-config "$VOLVE_LAYOUTS" --device cuda
    done
  done
done
```

For a resume, pass the exact cell's `latest.pt` through `--resume`; do not reuse
a checkpoint from another model, layout, or size.
