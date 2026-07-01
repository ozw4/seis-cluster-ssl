# F3 facies benchmark inspection

This runbook defines the shared config and artifact contract for inspecting the
F3 facies benchmark before SEGY conversion, embedding extraction, or downstream
facies experiments.

## Scope

This inspection stage records raw file inventory, SEGY geometry, PNG label
metadata, quicklook figures, label consistency checks, tokenization previews,
and a consolidated Japanese report/runbook output. It does not create formal
NPY volumes, manifests, embeddings, or training runs.

## Roots

- Raw data root: `/home/dcuser/data/public_data/field/F3`
- Artifact root: `/workspace/artifacts/seis_ssl_cluster`
- Inspection root: `/workspace/artifacts/seis_ssl_cluster/inspection/f3/facies_benchmark_v1`

Do not write F3 inspection outputs under `runs/`.

`artifacts/` is local generated output and is not tracked by Git. For GitHub
review, copy only lightweight inspection summaries and representative figures
to `results/f3/facies_benchmark_v1/inspection/`; do not commit raw data, SEGY,
full dumps, path lists, or other generated local artifacts.

## Config Contract

Every inspection YAML under
`experiments/f3/facies_benchmark_v1/00_inspection/` uses this shared top-level
shape:

```yaml
paths:
  f3_root: /home/dcuser/data/public_data/field/F3
  artifact_root: /workspace/artifacts/seis_ssl_cluster
outputs:
  inspection_dir: /workspace/artifacts/seis_ssl_cluster/inspection/f3/facies_benchmark_v1
dataset:
  name: f3_facies_benchmark
  version: facies_benchmark_v1
inspection:
  # Stage-specific inputs, outputs, and rendering controls.
```

The YAML must not include a top-level `stage`. The proc entrypoint selects the
stage and resolves the common contract.

## Output Layout

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

## Stages

| Order | Entrypoint | Config | Primary output |
|---|---|---|---|
| 1 | `inspect_f3_files.py` | `01_inspect_files.yaml` | `inventory/file_inventory.{json,csv}` |
| 2 | `inspect_f3_segy_geometry.py` | `02_inspect_segy_geometry.yaml` | `segy/segy_geometry.{json,csv}` |
| 3 | `inspect_f3_png_labels.py` | `03_inspect_png_labels.yaml` | `labels/png_label_inventory.{json,csv}` and `labels/facies_palette.json` |
| 4 | `visualize_f3_quicklook.py` | `04_make_quicklook_figures.yaml` | `quicklook/seismic/`, `quicklook/labels/`, and `quicklook/overlays/` |
| 5 | `check_f3_label_consistency.py` | `05_check_label_consistency.yaml` | `stats/label_consistency.{json,csv}` and `quicklook/consistency/` |
| 6 | `preview_f3_tokenization.py` | `06_make_tokenization_preview.yaml` | `quicklook/tokenization/` and `stats/tokenization_preview.json` |
| 7 | `build_f3_inspection_report.py` | `07_build_inspection_report.yaml` | `report.md` and `report.json` |

## Runbook

Run the inspection stages in order:

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

The final report links key figures by paths relative to the inspection root,
including `quicklook/seismic/seismic_xz_y_mid.png`,
`quicklook/overlays/train_inline_0250_overlay.png`, and
`quicklook/tokenization/train_inline_0250_tokenization.png`.

## Figure Contract

- Default PNG output uses `dpi: 200`; final figure candidates use `dpi: 300`.
- Tables and statistics should be written as CSV and JSON.
- Figures use a white background and restrained styling.
- Seismic amplitude quicklooks use a grayscale colormap.
- Facies labels use a fixed colorblind-friendly palette recorded in
  `labels/facies_palette.json`.
- Axis labels must identify inline, crossline, and sample or time/depth axes.
- XZ and YZ sections display increasing sample or time/depth downward.
- Comparison panels use matched extent, origin, aspect, and axes.
- Overlay alpha and amplitude clip percentiles are config fields and must be
  recorded in metadata JSON.
