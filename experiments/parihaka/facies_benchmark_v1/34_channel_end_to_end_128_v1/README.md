# Parihaka Channel larger-context end-to-end v1

This experiment repeats the paired pretrained-versus-random end-to-end
initialization study with a larger raw-amplitude context while retaining the
same supervised core. The encoder and decoder remain jointly trainable.

It has an independent output namespace and does not overwrite the earlier
[end-to-end experiment](../31_channel_end_to_end_v1/README.md). As in that
experiment, comparison with a frozen-embedding score does not isolate
fine-tuning because the input construction differs between regimes.

Experiment definition:
[larger-context YAML](01_channel_end_to_end_128.yaml). Study-wide context:
[Parihaka benchmark overview](../README.md).

Follow the phase order of the earlier end-to-end experiment. For this variant,
the matched device-feasibility pair is a hard gate: if either arm exceeds
capacity, stop without changing the fixed geometry or precision contract. Its
feasibility jobs occupy the same identities as their full jobs, so the full
pass recovers them instead of starting replacement jobs.
