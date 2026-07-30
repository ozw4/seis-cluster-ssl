# F3 XY-neighbour consensus hard-label review

- Target semantics: `xy_neighbor_consensus_hard_label_smoothing_v1`
- Training representation: `xy_neighbor_consensus_hard_labels_v1`
- Source hard manifest SHA-256: `c0b4edc17774d522bc0896db1366166628135d8369d2e063fd6183c66b01d15c`
- Consensus target manifest SHA-256: `307c9a28796f7b1d90ef4f188676abc5a3604be402fe8ef4a18b515d06671a41`
- Pretraining handoff SHA-256: `6f7faeac191c79285a5fddd397a3da5ea65ca69db83e69cf42f18aa3afa32500`

## Fixed exclusions

No embeddings, posterior tensors, affinities, emissions, Viterbi re-decoding, beta calibration, target refresh, or downstream labels/metrics enter this target contract.

## Temporal transition diagnostics

Temporal transition-count increases are allowed. These counts are observational diagnostics, not eligibility or stop gates.

## Head diagnostics

| K | Valid tokens | Changed tokens | Changed fraction | Source transitions | Output transitions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6 | 162960 | 4316 | 0.026485 | 26857 | 26287 |
| 8 | 162960 | 4477 | 0.027473 | 36375 | 35540 |
| 10 | 162960 | 4438 | 0.027234 | 37267 | 36391 |
