# Parihaka Channel benchmark v1

This experiment compares one pretrained Parihaka amplitude MAE with the same
architecture initialized randomly. The encoder embeddings are fixed; each job
trains only the same binary `VoxelDecoder3D`, with seed 42000 and all selected
section voxels used once per epoch.

Pretraining used the full unlabeled Parihaka amplitude volume, including
amplitudes from downstream validation and test sections. This benchmark
therefore measures survey-specific transductive self-supervised pretraining
benefit and does not establish inductive transfer to unseen surveys.

The repository does not choose section indices. First prepare and inspect the
labels, review `parihaka_channel_section_counts.csv`, copy
`02_layouts.example.yaml` outside its example name, and replace every placeholder
with explicit integer X/Y indices. Inline is fixed to the prepared volume's X axis
and crossline is fixed to its Y axis; this is a benchmark contract, not a layout
setting. The fixed sparse validation planes are shared by all 30 jobs and are
used only for checkpoint selection. A training line may not reuse a validation
line number in the same orientation. For each of small, medium, and large, all
five layouts must select different training-section sets; line order does not
make a set distinct.

Test supervision is also common to every layout and data size, but it is a
voxel-level complement, not a set of test sections. It contains valid-label,
valid-token voxels that belong to neither a validation plane nor any inline or
crossline in the union of all five layouts' large training candidates. Thus,
even for small and medium jobs, unused medium/large candidates and candidates
from other layouts remain excluded from test. Test is not the union of the
remaining inline and crossline planes. It is evaluated once from `best.pt`.

```bash
EXP=experiments/parihaka/facies_benchmark_v1/30_channel_benchmark_v1

PYTHONPATH=src python proc/seis_ssl_cluster/prepare_parihaka_channel_labels.py \
  --config "$EXP/01_prepare_channel_labels.yaml" --dry-run
PYTHONPATH=src python proc/seis_ssl_cluster/prepare_parihaka_channel_labels.py \
  --config "$EXP/01_prepare_channel_labels.yaml"
PYTHONPATH=src python proc/seis_ssl_cluster/inspect_parihaka_channel_sections.py \
  --config "$EXP/01_prepare_channel_labels.yaml"
```

Create the random checkpoint with the existing generic entrypoint, then run the
two extraction configs. They are identical except for checkpoint, model-tag path,
and output directory.

```bash
PYTHONPATH=src python proc/seis_ssl_cluster/create_random_mae_checkpoint.py \
  --config "$EXP/03_create_random_checkpoint.yaml"
PYTHONPATH=src python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/04_extract_pretrained_embeddings.yaml"
PYTHONPATH=src python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/05_extract_random_embeddings.yaml"
```

Run one read-only preflight, then the simple 30-job loop. Set `LAYOUT_CONFIG` to
the reviewed, concrete YAML rather than the example file.

```bash
LAYOUT_CONFIG=/absolute/path/to/parihaka_channel_layouts.yaml
PYTHONPATH=src python proc/seis_ssl_cluster/run_parihaka_channel_decoder.py \
  --config "$EXP/06_channel_benchmark.yaml" --model pretrained \
  --layout layout_000 --size small --layout-config "$LAYOUT_CONFIG" --dry-run

for model in pretrained random; do
  for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do
    for size in small medium large; do
      PYTHONPATH=src python proc/seis_ssl_cluster/run_parihaka_channel_decoder.py \
        --config "$EXP/06_channel_benchmark.yaml" --model "$model" \
        --layout "$layout" --size "$size" --layout-config "$LAYOUT_CONFIG"
    done
  done
done

PYTHONPATH=src python proc/seis_ssl_cluster/summarize_parihaka_channel_benchmark.py \
  --config "$EXP/06_channel_benchmark.yaml"
```

An interrupted job resumes only from its own `latest.pt` using `--resume PATH`.
Resume also requires the embedding/checkpoint metadata, label path, decoder and
training settings, fixed 8x8x8-core/1x1x1-halo tile settings, split class counts,
and tile counts to match the interrupted run exactly.
`best.pt` is selected by validation Channel IoU. Test is evaluated once after
training from `best.pt`; no probability volume is written. The summary requires
all 30 `metrics.json` files and reports paired test Channel-IoU deltas only.
Each metrics file carries the same benchmark identity used for resume, including
the seeded decoder-initial-state SHA-256 and the validated model-source role.
Preflight hashes the checkpoint files referenced by embedding metadata, requires
the pretrained source to be the benchmark's Parihaka `full_100ep/latest.pt`, and
requires the random source to carry the seed-42 `random_init` metadata referencing
that pretrained checkpoint and expected model tag. Before aggregation, the summary
requires common label, embedding metadata, geometry, decoder, training, and tile
identity across all jobs; one checkpoint and model-source identity within each
model; distinct pretrained/random checkpoint SHA-256 values; valid model roles;
and matching non-checkpoint identity, supervision, class weights, split counts,
and tile counts within every pair.
