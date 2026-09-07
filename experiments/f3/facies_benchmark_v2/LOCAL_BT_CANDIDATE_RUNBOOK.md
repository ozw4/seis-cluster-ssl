# F3 Local Barlow Twins candidate execution

This is the shared execution sequence for the Local Barlow Twins candidate
screens under experiments `119`–`123`. Each experiment README records only its
scientific question and result link; its YAML files define the candidate arms
and exact conditions.

Set the [shared F3 v2 environment](README.md), choose one matching pretraining,
embedding, and downstream configuration, and run from the repository root:

```bash
set -euo pipefail
: "${PRETRAIN_CONFIG:?select a pretraining YAML}"
: "${EMBEDDING_CONFIG:?select its embedding YAML}"
: "${DOWNSTREAM_CONFIG:?select its downstream YAML}"
export LAYOUT="${LAYOUT:-layout_001}"
export SIZE="${SIZE:-medium}"

python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$PRETRAIN_CONFIG" --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$PRETRAIN_CONFIG"

python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EMBEDDING_CONFIG" --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EMBEDDING_CONFIG" --skip-existing

python proc/seis_ssl_cluster/run_f3_lithology_candidate.py \
  --config "$DOWNSTREAM_CONFIG" --layout "$LAYOUT" --size "$SIZE" --dry-run
python proc/seis_ssl_cluster/run_f3_lithology_candidate.py \
  --config "$DOWNSTREAM_CONFIG" --layout "$LAYOUT" --size "$SIZE"
```

Repeat the downstream command for every layout and size required by the owning
experiment.

A continuation configuration must resume from the completed checkpoint for the
same arm and trajectory. Supply that checkpoint with `--resume` to both the
dry-run and live pretraining commands; do not run a continuation configuration
fresh or resume it from another arm.
