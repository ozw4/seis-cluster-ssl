# Parihaka Channel SSL/HMM comparison v1

This frozen-embedding experiment compares the MAE and global Barlow Twins
continuation controls with their source-matched HMM continuations. It asks
whether the stratigraphic second stage improves each representation under the
same Channel decoder protocol.

Experiment definition: [four-way YAML](05_channel_four_way.yaml). Study-wide
context: [Parihaka benchmark overview](../README.md). Continuations come from the
[SSL/HMM suite](../21_ssl_hmm_continuation_v1/README.md).

## Phase order

1. Confirm that every fixed-budget Stage 2 source is complete. Feasibility and
   validation-best checkpoints are not substitutes for those sources.
2. Dry-run and execute the explicit embedding extraction configurations in
   this directory.
3. Preflight a common decoder condition to verify paired supervision and
   decoder identity across sources.
4. Run the configured frozen-decoder matrix.
5. Audit completeness and write the summary only after every arm is present.
