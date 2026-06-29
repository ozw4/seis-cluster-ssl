# F3 facies benchmark v1

Inspection configs for the F3 facies benchmark before downstream few-label
facies evaluation.

Source-of-truth inputs:

- Raw F3 root: `/home/dcuser/data/public_data/field/F3`
- Artifact root: `/workspace/artifacts/seis_ssl_cluster`
- Inspection output: `/workspace/artifacts/seis_ssl_cluster/inspection/f3/facies_benchmark_v1`

Each YAML is intentionally standalone and avoids inheritance, anchors, merge
keys, and symlinks. Raw YAML does not contain a top-level `stage`; the selected
proc entrypoint owns the stage identity.

Inspection output layout:

```text
/workspace/artifacts/seis_ssl_cluster/inspection/f3/facies_benchmark_v1/
├── inventory/
├── segy/
├── labels/
├── quicklook/
│   ├── seismic/
│   ├── labels/
│   ├── overlays/
│   ├── consistency/
│   └── tokenization/
├── stats/
├── report.md
└── report.json
```

Run the inspection stages in this order:

```bash
python proc/seis_ssl_cluster/inspect_f3_files.py \
  --config experiments/f3/facies_benchmark_v1/00_inspection/01_inspect_files.yaml

python proc/seis_ssl_cluster/inspect_f3_segy_geometry.py \
  --config experiments/f3/facies_benchmark_v1/00_inspection/02_inspect_segy_geometry.yaml

python proc/seis_ssl_cluster/inspect_f3_png_labels.py \
  --config experiments/f3/facies_benchmark_v1/00_inspection/03_inspect_png_labels.yaml

python proc/seis_ssl_cluster/visualize_f3_quicklook.py \
  --config experiments/f3/facies_benchmark_v1/00_inspection/04_make_quicklook_figures.yaml

python proc/seis_ssl_cluster/check_f3_label_consistency.py \
  --config experiments/f3/facies_benchmark_v1/00_inspection/05_check_label_consistency.yaml

python proc/seis_ssl_cluster/preview_f3_tokenization.py \
  --config experiments/f3/facies_benchmark_v1/00_inspection/06_make_tokenization_preview.yaml

python proc/seis_ssl_cluster/build_f3_inspection_report.py \
  --config experiments/f3/facies_benchmark_v1/00_inspection/07_build_inspection_report.yaml
```

See [docs/f3_facies_benchmark_inspection.md](../../../docs/f3_facies_benchmark_inspection.md)
for the shared config contract and figure conventions.
