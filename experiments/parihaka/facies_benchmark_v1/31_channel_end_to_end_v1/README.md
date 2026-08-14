# Parihaka Channel end-to-end v1

This experiment separates two paired scientific questions:

- **Frozen representation:** frozen pretrained embeddings versus frozen random
  embeddings, with only `VoxelDecoder3D` trained.
- **End-to-end initialization:** a trainable encoder initialized from the
  pretrained checkpoint (`finetune_pretrained`) versus the same trainable
  encoder initialized from the seed-42 random checkpoint
  (`train_from_scratch`).

The MAE was pretrained on the full unlabeled Parihaka amplitude volume,
including amplitudes at downstream validation and test sections. Both
comparisons are therefore survey-specific, transductive comparisons; they do
not establish inductive transfer to an unseen survey.

The fixed sparse validation inline/crossline planes remain checkpoint-selection
only. Test is identical across all layouts, sizes, and frozen/end-to-end
conditions: valid-label, valid-token voxels outside validation and outside the
union of every inline and crossline in all five layouts' large training
candidates. This is a voxel-level complement, not the union of remaining
inline/crossline planes. Consequently, small and medium jobs do not test on
their unused large candidates or on another layout's candidates. Test is
evaluated once from the validation-selected best checkpoint.

Within the end-to-end pair, encoder initialization is the only changed
condition. Decoder initialization, supervision, tile order, optimizer,
training settings, and runtime precision contract are paired. The frozen and
end-to-end pairs answer different questions. Frozen jobs use 128^3 overlap
embedding context, while end-to-end jobs use an 80^3 raw-amplitude encoder
crop. A score difference between regimes therefore includes an input-context
difference and must not be described as the isolated effect of fine-tuning.

Set these paths before running. `LAYOUT_CONFIG` must be the reviewed concrete
layout YAML, not the example with placeholders. Scientific jobs explicitly use
CUDA; `auto` is not used below.

```bash
EXP=experiments/parihaka/facies_benchmark_v1/31_channel_end_to_end_v1
CONFIG="$EXP/01_channel_end_to_end.yaml"
FROZEN_CONFIG=experiments/parihaka/facies_benchmark_v1/30_channel_benchmark_v1/06_channel_benchmark.yaml
LAYOUT_CONFIG=/absolute/path/to/parihaka_channel_layouts.yaml
RUNS_ROOT="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/channel_end_to_end/runs"
```

## Preflight all 15 paired conditions

This runs both encoder initializations for every layout/size pair and writes
nothing.

```bash
for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do
  for size in small medium large; do
    for encoder_init in pretrained random; do
      PYTHONPATH=src python proc/seis_ssl_cluster/run_parihaka_channel_end_to_end.py \
        --config "$CONFIG" --encoder-init "$encoder_init" \
        --layout "$layout" --size "$size" \
        --layout-config "$LAYOUT_CONFIG" --device cuda --dry-run
    done
  done
done
```

## Paired two-step smoke and resume

Interrupt both `layout_003/small` jobs after two optimizer steps:

```bash
for encoder_init in pretrained random; do
  PYTHONPATH=src python proc/seis_ssl_cluster/run_parihaka_channel_end_to_end.py \
    --config "$CONFIG" --encoder-init "$encoder_init" \
    --layout layout_003 --size small --layout-config "$LAYOUT_CONFIG" \
    --device cuda --max-steps 2
done
```

Resume each smoke checkpoint through validation-best selection and the single
test evaluation:

```bash
for encoder_init in pretrained random; do
  JOB_DIR="$RUNS_ROOT/encoder_init=$encoder_init/layout=layout_003/size=small"
  PYTHONPATH=src python proc/seis_ssl_cluster/run_parihaka_channel_end_to_end.py \
    --config "$CONFIG" --encoder-init "$encoder_init" \
    --layout layout_003 --size small --layout-config "$LAYOUT_CONFIG" \
    --device cuda --resume "$JOB_DIR/latest.pt"
done
```

## Complete all 30 jobs

For a fresh artifact root, the direct loop is:

```bash
for encoder_init in pretrained random; do
  for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do
    for size in small medium large; do
      PYTHONPATH=src python proc/seis_ssl_cluster/run_parihaka_channel_end_to_end.py \
        --config "$CONFIG" --encoder-init "$encoder_init" \
        --layout "$layout" --size "$size" \
        --layout-config "$LAYOUT_CONFIG" --device cuda
    done
  done
done
```

For restartable sequential execution, skip completed jobs and resume only an
existing `latest.pt`:

```bash
for encoder_init in pretrained random; do
  for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do
    for size in small medium large; do
      JOB_DIR="$RUNS_ROOT/encoder_init=$encoder_init/layout=$layout/size=$size"
      if [ -f "$JOB_DIR/metrics.json" ]; then
        echo "completed: $encoder_init/$layout/$size"
      elif [ -f "$JOB_DIR/latest.pt" ]; then
        PYTHONPATH=src python proc/seis_ssl_cluster/run_parihaka_channel_end_to_end.py \
          --config "$CONFIG" --encoder-init "$encoder_init" \
          --layout "$layout" --size "$size" \
          --layout-config "$LAYOUT_CONFIG" --device cuda \
          --resume "$JOB_DIR/latest.pt"
      else
        PYTHONPATH=src python proc/seis_ssl_cluster/run_parihaka_channel_end_to_end.py \
          --config "$CONFIG" --encoder-init "$encoder_init" \
          --layout "$layout" --size "$size" \
          --layout-config "$LAYOUT_CONFIG" --device cuda
      fi
    done
  done
done
```

Require exactly 30 completed metric files before aggregation:

```bash
METRICS_COUNT=$(find "$RUNS_ROOT" -type f -name metrics.json | wc -l)
test "$METRICS_COUNT" -eq 30
```

Write the end-to-end paired summary only after all 30 jobs complete:

```bash
PYTHONPATH=src python proc/seis_ssl_cluster/summarize_parihaka_channel_end_to_end.py \
  --config "$CONFIG" --dry-run
PYTHONPATH=src python proc/seis_ssl_cluster/summarize_parihaka_channel_end_to_end.py \
  --config "$CONFIG"
```

After both the frozen 30 jobs and end-to-end 30 jobs are complete, write the
four-condition descriptive table:

```bash
PYTHONPATH=src python proc/seis_ssl_cluster/summarize_parihaka_channel_four_way.py \
  --config "$CONFIG" --frozen-config "$FROZEN_CONFIG" --dry-run
PYTHONPATH=src python proc/seis_ssl_cluster/summarize_parihaka_channel_four_way.py \
  --config "$CONFIG" --frozen-config "$FROZEN_CONFIG"
```

The four-condition report presents `frozen_pretrained - frozen_random` and
`finetune_pretrained - train_from_scratch` as separate paired deltas. It does
not compute or name a cross-regime fine-tuning delta.
