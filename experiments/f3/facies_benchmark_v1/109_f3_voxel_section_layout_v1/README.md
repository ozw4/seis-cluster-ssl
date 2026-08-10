# F3 voxel section-layout benchmark v1

This directory preregisters the closed model roster for the section-layout
downstream benchmark. `00_model_roster.yaml` resolves exactly 14 existing
frozen-embedding sources. It does not discover models from directories or file
names, and it does not authorize pretraining or embedding extraction.

The user-run calibration tool writes a separate
`f3_voxel_section_layout_contract_v1` mapping containing five concrete layouts.
Each layout must contain nested small, medium, and large selections with 1+1,
2+2, and 4+4 inline/crossline teacher sections. It will also copy the three
fixed integer target voxel counts calculated from historical cap-dataset
medians. Benchmark builders and runners consume only that generated contract;
they must not read an old cap manifest.

Run candidate inspection first:

```bash
PYTHONPATH=src python proc/seis_ssl_cluster/prepare_f3_lithology_voxel_section_layout_contract.py \
  --config experiments/f3/facies_benchmark_v1/109_f3_voxel_section_layout_v1/01_prepare_section_layout_contract.yaml \
  --mode inspect --dry-run
```

Remove `--dry-run` to write the candidate CSV/JSON. Copy
`02_layout_lines.example.yaml` to the configured `02_layout_lines.yaml`, replace
the example numbers with five reviewed 4+4 train layouts, and run
`--mode finalize` (with `--dry-run` first). Finalize writes the canonical
contract only after every class, monitored-class, line-contribution, nesting,
validation-identity, and target-error gate passes. Neither mode reads model
metrics, checkpoints, or embeddings.

The statistical unit is `layout_id`, the validation mask is shared across all
jobs, and every decoder uses seed 42000. Models with `selection_role:
diagnostic` produce metrics but are ineligible for formal adoption.

This directory contains no runner config. The calibration command is not a
scientific model job; creating benchmark datasets, training decoders, running
inference, or producing a summary remains outside this issue.
