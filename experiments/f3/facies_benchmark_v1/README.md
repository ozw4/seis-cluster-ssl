# F3 facies benchmark v1 artifact producers

This historical namespace retains reusable F3 inspection, preparation,
embedding, pseudo-target, pretraining, and validation producers. It is not the
current benchmark-result workflow. The F3-native MAE / Barlow Twins continuation
suite is documented in
[21_ssl_hmm_continuation_v1](21_ssl_hmm_continuation_v1/README.md).

The inspection stages must run in numeric order before prepared data is used by
later experiments. Their YAML files own paths, rendering options, and output
names.

```bash
export EXP=experiments/f3/facies_benchmark_v1

python proc/seis_ssl_cluster/inspect_f3_files.py \
  --config "$EXP/00_inspection/01_inspect_files.yaml"
python proc/seis_ssl_cluster/inspect_f3_segy_geometry.py \
  --config "$EXP/00_inspection/02_inspect_segy_geometry.yaml"
python proc/seis_ssl_cluster/inspect_f3_png_labels.py \
  --config "$EXP/00_inspection/03_inspect_png_labels.yaml"
python proc/seis_ssl_cluster/visualize_f3_quicklook.py \
  --config "$EXP/00_inspection/04_make_quicklook_figures.yaml"
python proc/seis_ssl_cluster/check_f3_label_consistency.py \
  --config "$EXP/00_inspection/05_check_label_consistency.yaml"
python proc/seis_ssl_cluster/preview_f3_tokenization.py \
  --config "$EXP/00_inspection/06_make_tokenization_preview.yaml"
```

Review geometry, label orientation, aligned overlays, and tokenization previews
before continuing.
