# F3 lithology MAE / Local Barlow Twins five-way v2

This directory has two roles. Its prepared volume and encoder extractions are
shared inputs to the current v3 comparison. Its positional section-layout
comparison, defined by [60_five_way.yaml](60_five_way.yaml), is historical.
Upstream encoder training remains in the
[v1 five-way workflow](../../facies_benchmark_v1/110_lithology_mae_local_bt_five_way_v1/README.md).

## Shared prepared sources

After any required inspection, prepare the volume, reference token geometry,
and canonical voxel supervision. Then pass the prepared-volume parity gate and
extract each five-way source. These are the only phases here required by v3.

```bash
set -euo pipefail
export PREP=experiments/f3/facies_benchmark_v2/10_prepare
export EXP=experiments/f3/facies_benchmark_v2/110_lithology_mae_local_bt_five_way_v2

python proc/seis_ssl_cluster/prepare_f3_facies_volume.py \
  --config "$PREP/01_prepare_f3_volume.yaml" --dry-run
python proc/seis_ssl_cluster/prepare_f3_facies_volume.py \
  --config "$PREP/01_prepare_f3_volume.yaml"
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$PREP/02_extract_reference_valid_tokens.yaml" --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$PREP/02_extract_reference_valid_tokens.yaml"
python proc/seis_ssl_cluster/build_f3_lithology_voxel_dataset.py \
  --config "$PREP/03_build_voxel_supervision.yaml" --dry-run
python proc/seis_ssl_cluster/build_f3_lithology_voxel_dataset.py \
  --config "$PREP/03_build_voxel_supervision.yaml"

python proc/seis_ssl_cluster/check_f3_prepared_volume_parity.py --dry-run
python proc/seis_ssl_cluster/check_f3_prepared_volume_parity.py

for config in "$EXP"/50_embeddings/*.yaml; do
  python proc/seis_ssl_cluster/extract_embeddings.py --config "$config" --dry-run
  python proc/seis_ssl_cluster/extract_embeddings.py --config "$config"
done
```

## Historical v2 downstream

To reproduce the positional-layout comparison, first build the
[v2 layout](../109_f3_voxel_section_layout_v2/README.md), then apply the source
audit, cell execution, and summary sequence from the
[current v3 runbook](../110_lithology_mae_local_bt_five_way_v3/README.md) using
this directory's `60_five_way.yaml`. Do not label those results as v3.

## Recovery

The common runner's recovery rules are owned by the
[current v3 runbook](../110_lithology_mae_local_bt_five_way_v3/README.md#再開).
