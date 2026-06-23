# Seismic SSL Cluster Runbook

Use the YAML files in `proc/configs/seis_ssl_cluster/` as raw user configs.
They are intentionally minimal; each proc entrypoint resolves them into a full
runtime config before execution. See `docs/configuration.md` for the complete
ownership table, migration notes, and YAML examples.

## Handoff Policy

This pipeline uses Option 1: explicit paths in every stage YAML. Do not rely on
automatic derivation from dataset, version, or run IDs. Every downstream config
must name the upstream artifact path it consumes.

| Step | Entrypoint | Config | Explicit handoff |
|---|---|---|---|
| Build manifest | `build_nopims_manifests.py` | `paths`, `manifest` | Reads `manifest.input_path_list`; writes `manifest.output_dir` / `manifest.output_name` |
| Normalization stats | `prepare_nopims_normalization_stats.py` | `paths`, `manifests`, `normalization` | Reads `manifests.train`; writes per-survey stat paths recorded in the manifest |
| Normalization QC | `filter_manifest_by_normalization_qc.py` | `paths`, `manifests`, `splits`, `qc` | Reads original manifest and split; writes clean manifest, clean split, and QC outputs |
| MAE training | `train_amp_mae.py` | `paths`, `manifests`, `data`, `zero_mask`, `model`, `masking`, `loss`, `train`, `visualization` | Reads clean manifest and clean path-list; writes `resolved_config.json` and checkpoints |
| Embedding extraction | `extract_embeddings.py` | `paths`, `manifests`, `embeddings`, `embedding` | Reads clean manifest and checkpoint; writes embedding artifacts |
| Clustering | `cluster_embeddings.py` | `paths`, `embeddings`, `clustering` | Reads embedding directory; writes clustering models and labels |
| Visualization | `visualize_clusters.py` | `paths`, `clustering`, `visualization` | Reads clustering directory; writes PNGs, summaries, and optional voxel labels |

## Commands

Run a dry-run before each execution. Dry-runs print only the settings relevant
to the selected stage.

```bash
python proc/seis_ssl_cluster/build_nopims_manifests.py \
  --config proc/configs/seis_ssl_cluster/build_nopims_manifests.yaml \
  --dry-run

python proc/seis_ssl_cluster/prepare_nopims_normalization_stats.py \
  --config proc/configs/seis_ssl_cluster/prepare_nopims_normalization_stats.yaml \
  --dry-run

python proc/seis_ssl_cluster/filter_manifest_by_normalization_qc.py \
  --config proc/configs/seis_ssl_cluster/filter_manifest_by_normalization_qc.yaml \
  --dry-run

python proc/seis_ssl_cluster/train_amp_mae.py \
  --config proc/configs/seis_ssl_cluster/train_amp_mae.yaml \
  --device cuda \
  --max-steps 2 \
  --output-root /workspace/artifacts/seis_ssl_cluster/runs/smoke_amp_mae

python proc/seis_ssl_cluster/extract_embeddings.py \
  --config proc/configs/seis_ssl_cluster/extract_embeddings.yaml \
  --device cuda

python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config proc/configs/seis_ssl_cluster/cluster_embeddings.yaml

python proc/seis_ssl_cluster/visualize_clusters.py \
  --config proc/configs/seis_ssl_cluster/visualize_clusters.yaml
```

## Operational Notes

Raw YAML must not include a top-level `stage`. The selected proc script owns the
stage identity.

Fixed amplitude-only fields such as `data.grid_order`, model channel counts,
`masking.spatial_mask_mode`, and `loss.valid_mask_mode` are code-owned and
appear only in the resolved config. Training YAML owns `loss.reconstruction`
and must set it to `huber`, `mse`, or `l1`.

Embedding extraction is checkpoint-owned for model, zero-mask, masking, and loss
settings. Configure only the checkpoint path, clean manifest, extraction
geometry, and embedding output path.

The visualization default is safe for repeated runs: token maps and summaries
are enabled, while voxel reconstruction is disabled unless
`visualization.reconstruct_voxel` is set to `true` and intended surveys are
selected explicitly.
