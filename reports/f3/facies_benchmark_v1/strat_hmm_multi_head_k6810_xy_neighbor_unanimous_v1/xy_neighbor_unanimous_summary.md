# F3 unanimous XY-neighbour hard-label review

- Target semantics: `xy_neighbor_unanimous_outlier_correction_v1`
- Target representation: `xy_neighbor_unanimous_hard_labels_v1`
- Target audit status: `XYUNANIM_TARGET_GO`
- Target manifest SHA-256: `f09cc73c67e686785d6b04dfb4b92aea4a339a234d71226cd317e1380323d1a4`
- Pretraining handoff SHA-256: `892bc9dd3f19250730f0f3c6474923caa81211a0bd48cc38abc84bafb7f70ce1`

| K | Valid tokens | Changed tokens | Changed fraction | Source transitions | Output transitions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6 | 162960 | 1080 | 0.006627 | 26857 | 26708 |
| 8 | 162960 | 1031 | 0.006327 | 36375 | 36171 |
| 10 | 162960 | 1086 | 0.006664 | 37267 | 37022 |

No posterior tensors, lateral smoothing, Viterbi re-decoding, or target-refresh evidence is part of this review.
