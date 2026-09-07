# Parihaka Channel MAE/local-BT comparison v1

This experiment compares MAE and local Barlow Twins continuation controls with
their source-matched HMM continuations under the frozen Channel decoder
protocol. Its scientific delta from the preceding comparison is the local
Barlow Twins branch.

The MAE embeddings and decoder jobs are reused in place from the
[global SSL/HMM comparison](../32_channel_ssl_hmm_four_way_v1/README.md).
Only the local-BT embeddings and local-BT decoder jobs are new; existing MAE
artifacts must not be copied, re-extracted, or rerun.

Experiment definition:
[combined comparison YAML](03_channel_mae_local_bt_four_way.yaml). Study-wide
context: [Parihaka benchmark overview](../README.md).

Follow the extraction, preflight, decoder, and summary ordering in the
preceding comparison, but execute only the local-BT arms. Confirm the reused
MAE inputs first and aggregate the complete reused-plus-new matrix last. A
missing or incompatible reused input is a failed prerequisite, not permission
to regenerate it in this phase.
