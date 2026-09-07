# Parihaka Channel end-to-end v1

This experiment compares a trainable encoder initialized from the pretrained
source with the same encoder initialized randomly. Encoder initialization is
the paired variable; encoder and decoder are trained together.

The frozen comparison in the
[Channel baseline](../30_channel_benchmark_v1/README.md) answers a different
question. Frozen evaluation consumes offline, overlap-aggregated full-volume
embeddings, while this experiment encodes supervised raw-amplitude tiles during
training. A cross-regime score difference must therefore not be described as
the isolated effect of fine-tuning.

Study-wide context is in the
[Parihaka benchmark overview](../README.md). Experiment definition:
[end-to-end YAML](01_channel_end_to_end.yaml).

## Phase order

1. Dry-run every paired condition.
2. Run a matched short feasibility pair.
3. Complete the configured matrix.
4. Write the paired end-to-end summary after the matrix is complete.
5. Produce a four-condition descriptive comparison only after both this
   experiment and the frozen baseline are complete.

The optional four-condition view must keep the frozen and end-to-end paired
deltas separate; it does not define a cross-regime fine-tuning delta.
