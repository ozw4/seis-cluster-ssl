# Parihaka Channel benchmark v1

This experiment measures the benefit of a frozen Parihaka-pretrained amplitude
encoder against the same architecture initialized randomly. Both embedding
volumes are fixed; only the common binary Channel decoder is trained.

Study-wide context is in the
[Parihaka benchmark overview](../README.md). Experiment definitions:
[benchmark configuration](06_channel_benchmark.yaml) and reviewed
[layout definition](02_layouts.yaml).

## Phase order

1. Prepare and inspect Channel labels with
   [the preparation config](01_prepare_channel_labels.yaml), then review the
   concrete layout definition. Layout review is a human prerequisite because
   the repository does not choose geological sections.
2. Create the architecture-matched random source, then extract pretrained and
   random embeddings from their explicit YAML configurations.
3. Dry-run the paired decoder matrix and confirm that both sources resolve to
   the same supervision and decoder identity.
4. Run the frozen-decoder jobs and aggregate only after every configured pair
   is complete.

Outputs created under an older supervision contract are incompatible and must
be isolated before a fresh run rather than silently resumed.
