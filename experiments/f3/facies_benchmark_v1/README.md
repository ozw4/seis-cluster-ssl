# F3 facies benchmark v1 artifact producers

This directory contains reusable F3 artifact-production stages. Active stages
cover raw-data inspection and preparation, embedding and target generation,
stratigraphic clustering, pretraining, extraction, and downstream-independent
validation. It is not a benchmark result-generation surface.

Source-of-truth inputs:

```bash
ROOT=/workspace/artifacts/seis_ssl_cluster
EXP=experiments/f3/facies_benchmark_v1
```

- Raw F3 root: `/home/dcuser/data/public_data/field/F3`
- Artifact root: `$ROOT`
- Inspection output: `$ROOT/inspection/f3/facies_benchmark_v1`

Each YAML is standalone and avoids inheritance, anchors, merge keys, and
symlinks. Raw YAML does not contain a top-level `stage`; the selected proc
entrypoint owns the stage identity.

Run the inspection stages in order:

```bash
python proc/seis_ssl_cluster/inspect_f3_files.py --config $EXP/00_inspection/01_inspect_files.yaml
python proc/seis_ssl_cluster/inspect_f3_segy_geometry.py --config $EXP/00_inspection/02_inspect_segy_geometry.yaml
python proc/seis_ssl_cluster/inspect_f3_png_labels.py --config $EXP/00_inspection/03_inspect_png_labels.yaml
python proc/seis_ssl_cluster/visualize_f3_quicklook.py --config $EXP/00_inspection/04_make_quicklook_figures.yaml
python proc/seis_ssl_cluster/check_f3_label_consistency.py --config $EXP/00_inspection/05_check_label_consistency.yaml
python proc/seis_ssl_cluster/preview_f3_tokenization.py --config $EXP/00_inspection/06_make_tokenization_preview.yaml
```

See [the inspection contract](../../../docs/f3_facies_benchmark_inspection.md)
for the shared config contract and figure conventions.
